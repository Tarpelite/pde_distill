import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# import torchvision.utils
from tqdm import tqdm
from utils import get_dataset, get_network, evaluate_synset, get_time
# import wandb
import copy
import random
from reparam_module import ReparamModule

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from omegaconf import DictConfig, OmegaConf
import hydra


def mse_loss_complex(output, target):
    loss_real = torch.nn.functional.mse_loss(output.real, target.real, reduction="sum")
    loss_imag = torch.nn.functional.mse_loss(output.imag, target.imag, reduction="sum")
    return loss_real + loss_imag

def init_syn_data(args, train_loader):
    real_example = next(iter(train_loader))

    assert args.num_channel == real_example[1].shape[-1]
    assert args.num_t <= real_example[1].shape[-2]
    # assert args.grid_size <= real_example[1].shape[2]
    with torch.no_grad():
        if args.syn_data_init == 'real':
            pde_data_sync = real_example[1].to(args.device)
            for input_data, target_data, grid in train_loader:
                pde_data_sync = torch.cat((pde_data_sync, target_data.to(args.device)), dim=0)
                if pde_data_sync.shape[0] >= args.sync_num:
                    pde_data_sync = pde_data_sync[:args.sync_num]
                    break
        else:
            print('initialize synthetic data from random noise')
            pde_data_sync = torch.randn(size=(args.sync_num, real_example[1].shape[1], real_example[1].shape[-2], real_example[1].shape[-1])).to(args.device)
            # print('======pde_data_sync shape: ', pde_data_sync.shape)
            
    syn_grid = real_example[-1][0].detach().to(args.device).repeat(args.sync_num, 1, 1)
    pde_data_sync = pde_data_sync.detach().to(args.device).requires_grad_(True)

    return pde_data_sync, syn_grid

def get_optimizer(args):
    if args.optimizer == "adam":
        return torch.optim.Adam
    elif args.optimizer == "sgd":
        return torch.optim.SGD
    else:
        raise AssertionError("Optimizer not recognized")

def load_buffer(args):
    expert_dir = os.path.join(args.buffer_path, args.filename)
    print("Expert Dir: {}".format(expert_dir))
    expert_files = []
    if args.load_all:
        buffer = []
        n = 0
        while os.path.exists(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n))):
            buffer = buffer + torch.load(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n)))
            n += 1
        if n == 0:
            raise AssertionError("No buffers detected at {}".format(expert_dir))

    else:
        n = 0
        while os.path.exists(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n))):
            expert_files.append(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n)))
            n += 1
        if n == 0:
            raise AssertionError("No buffers detected at {}".format(expert_dir))
        file_idx = 0
        expert_idx = 0
        random.shuffle(expert_files)
        if args.max_files is not None:
            expert_files = expert_files[:args.max_files]
        print("loading file {}".format(expert_files[file_idx]))
        buffer = torch.load(expert_files[file_idx])
        if args.max_experts is not None:
            buffer = buffer[:args.max_experts]
        random.shuffle(buffer)
    return buffer, expert_files 



def evaluate_syn_data(args, pde_data_sync, syn_grid, model_name_eval, syn_lr, test_loader, best_loss, best_std, save_this_it, it):
    print('-------------------------\nEvaluation\nmodel_train = %s, model_name_eval = %s, iteration = %d'%(args.model_name, model_name_eval, it))
    losss_test = []
    losss_train = []
    for it_eval in range(args.num_eval):
        net_eval = get_network(args).to(args.device) # get a random model         
        pde_data_sync_eval = copy.deepcopy(pde_data_sync.detach()) # avoid any unaware modification

        _, loss_train, loss_test = evaluate_synset(it_eval, net_eval, pde_data_sync_eval, syn_grid, test_loader, syn_lr.item(), args)
        losss_test.append(loss_test)
        losss_train.append(loss_train)
    losss_test = np.array(losss_test)
    losss_train = np.array(losss_train)
    loss_test_mean = np.mean(losss_test)
    loss_test_std = np.std(losss_test)
    if loss_test_mean > best_loss[model_name_eval]:
        best_loss[model_name_eval] = loss_test_mean
        best_std[model_name_eval] = loss_test_std
        save_this_it = True
    print('Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------'%(len(losss_test), model_name_eval, loss_test_mean, loss_test_std))
    # wandb.log({'Accuracy/{}'.format(model_name_eval): loss_test_mean}, step=it)
    # wandb.log({'Max_Accuracy/{}'.format(model_name_eval): best_loss[model_name_eval]}, step=it)
    # wandb.log({'Std/{}'.format(model_name_eval): loss_test_std}, step=it)
    # wandb.log({'Max_Std/{}'.format(model_name_eval): best_std[model_name_eval]}, step=it)
    
def save_syn_data(pde_data_sync, it, save_this_it, args):
    with torch.no_grad():
        pde_save = pde_data_sync.cuda()
        save_dir = os.path.join(".", "logged_files", args.filename)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(pde_save.cpu(), os.path.join(save_dir, "pdes_{}.pt".format(it)))
        if save_this_it:
            torch.save(pde_save.cpu(), os.path.join(save_dir, "pdes_best.pt".format(it)))
        # wandb.log({"Pixels": wandb.Histogram(torch.nan_to_num(pde_data_sync.detach().cpu()))}, step=it)


def train_syn_data_step(args, buffer, pde_data_sync, syn_grid, syn_lr, expert_files, criterion, optimizer_pde, optimizer_lr, it):
    student_net = get_network(args).to(args.device)  # get a random model
    student_net = ReparamModule(student_net)
    if args.distributed:
        student_net = torch.nn.DataParallel(student_net)
    student_net.train()
    # for name, param in student_net.named_parameters():
    #     print(name, param.shape)
    # exit()
    num_params = sum([np.prod(p.size()) for p in (student_net.parameters())])

    if args.load_all:
        expert_trajectory = buffer[np.random.randint(0, len(buffer))]
    else:
        expert_trajectory = buffer[expert_idx]
        expert_idx += 1
        if expert_idx == len(buffer):
            expert_idx = 0
            file_idx += 1
            if file_idx == len(expert_files):
                file_idx = 0
                random.shuffle(expert_files)
            print("loading file {}".format(expert_files[file_idx]))
            if args.max_files != 1:
                del buffer
                buffer = torch.load(expert_files[file_idx])
            if args.max_experts is not None:
                buffer = buffer[:args.max_experts]
            random.shuffle(buffer)

    start_epoch = np.random.randint(10, args.max_start_epoch)
    starting_params = expert_trajectory[start_epoch]

    target_params = expert_trajectory[start_epoch+args.expert_epochs]
    target_params = torch.cat([p.data.to(args.device).reshape(-1) for p in target_params], 0)

    student_params = [torch.cat([p.data.to(args.device).reshape(-1) for p in starting_params], 0).requires_grad_(True)]

    starting_params = torch.cat([p.data.to(args.device).reshape(-1) for p in starting_params], 0)

    syn_pdes = pde_data_sync

    param_loss_list = []
    param_dist_list = []
    indices_chunks = []

    for step in range(args.syn_steps):
        if not indices_chunks:
            indices = torch.randperm(len(syn_pdes))
            indices_chunks = list(torch.split(indices, args.batch_syn))
        these_indices = indices_chunks.pop()

        x = syn_pdes[these_indices][..., :-1, :]
        this_y = syn_pdes[these_indices][..., -1, :].unsqueeze(-2)

        if args.distributed:
            forward_params = student_params[-1].unsqueeze(0).expand(torch.cuda.device_count(), -1)
        else:
            forward_params = student_params[-1]

        x = student_net(x.reshape(x.shape[0], x.shape[1], -1), syn_grid[:x.shape[0]], flat_param=forward_params)
        ce_loss = criterion(x, this_y)

        grad = torch.autograd.grad(ce_loss, student_params[-1], create_graph=True)[0]
        grad = torch.complex((torch.clamp(grad.real, -1, 1)), (torch.clamp(grad.imag, -1, 1)))
        student_params.append(student_params[-1] - syn_lr * grad)

    param_loss = torch.tensor(0.0).to(args.device)
    param_dist = torch.tensor(0.0).to(args.device)

    param_loss += mse_loss_complex(student_params[-1], target_params)
    param_dist += mse_loss_complex(starting_params, target_params)
    diff = student_params[-1] - target_params
    diff = diff.abs()
    
    param_loss_list.append(param_loss)
    param_dist_list.append(param_dist)

    param_loss /= num_params
    param_dist /= num_params


    print('========== %d ->%d param_loss = %.6f, param_dist = %.6f'%(start_epoch, start_epoch+args.expert_epochs, param_loss.item(), param_dist.item()))
    param_loss /= param_dist

    grand_loss = param_loss

    optimizer_pde.zero_grad()
    optimizer_lr.zero_grad()

    grand_loss.backward()

    optimizer_pde.step()
    optimizer_lr.step()

    # wandb.log({"Grand_Loss": grand_loss.detach().cpu(),
    #            "Start_Epoch": start_epoch})

    for _ in student_params:
        del _

    if it%5 == 0:
        print('%s train syn_data iter = %04d, %d ->%d, grand_loss = %.4f' % (get_time(), it, start_epoch, start_epoch+args.expert_epochs, grand_loss.item()))



@hydra.main(version_base=None, config_path=".", config_name="distill_1DCFD")
def main(args):
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Hyper-parameters: \n', args.__dict__)

    train_loader, test_loader = get_dataset(args.data_dir, args.filename, args.batch_real, num_workers=args.num_workers, args=args)

    buffer, expert_files = load_buffer(args)

    # initialize the synthetic data and syn_lr
    pde_data_sync, syn_grid = init_syn_data(args, train_loader)
    syn_lr = torch.tensor(args.lr_teacher).to(args.device).requires_grad_(True)

    # optimizer_pde = torch.optim.SGD([pde_data_sync], lr=args.lr_pde, momentum=0.5)
    optimizer_pde = torch.optim.Adam([pde_data_sync], lr=args.lr_pde)
    optimizer_lr = torch.optim.SGD([syn_lr], lr=args.lr_lr, momentum=0.5)

    criterion = nn.MSELoss().to(args.device)
    
    best_loss = {m: 0 for m in args.model_eval_pool}
    best_std = {m: 0 for m in args.model_eval_pool}

    for it in range(0, args.Iteration+1):
        save_this_it = False

        # writer.add_scalar('Progress', it, it)
        # wandb.log({"Progress": it}, step=it)
        
        ''' Evaluate synthetic data '''
        if it % args.eval_it == 0:
            for model_name_eval in args.model_eval_pool:
                evaluate_syn_data(args, pde_data_sync, syn_grid, model_name_eval, syn_lr, test_loader, best_loss, best_std, save_this_it, it)

            if save_this_it or it % 1000 == 0:
                save_syn_data(pde_data_sync, it, save_this_it, args)

        # wandb.log({"Synthetic_LR": syn_lr.detach().cpu()}, step=it)

        train_syn_data_step(args, buffer, pde_data_sync, syn_grid, syn_lr, expert_files, criterion, optimizer_pde, optimizer_lr, it)

    # wandb.finish()


if __name__ == '__main__':
    main()
