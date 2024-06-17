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

@hydra.main(version_base=None, config_path=".", config_name="distill_1DCFD")
def main(args):
    # if args.max_experts is not None and args.max_files is not None:
    #     args.total_experts = args.max_experts * args.max_files

    print("CUDNN STATUS: {}".format(torch.backends.cudnn.enabled))

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    eval_it_pool = np.arange(0, args.Iteration + 1, args.eval_it).tolist()
    train_loader, test_loader = get_dataset(args.data_dir, args.filename, args.batch_real, num_workers=args.num_workers, args=args)
    model_eval_pool = ["FNO1D"]

    args.distributed = torch.cuda.device_count() > 1

    print('Hyper-parameters: \n', args.__dict__)
    print('Evaluation model pool: ', model_eval_pool)

    ''' initialize the synthetic data '''

    real_example = next(iter(train_loader))
    assert args.num_channel == real_example[1].shape[-1]
    assert args.num_t <= real_example[1].shape[-2]
    # assert args.grid_size <= real_example[1].shape[2]

    if args.syn_data_init == 'real':
        pass
    else:
        print('initialize synthetic data from random noise')
        pde_data_sync = torch.randn(size=(args.sync_num, real_example[1].shape[1], real_example[1].shape[-2], real_example[1].shape[-1]))
        syn_grid = real_example[-1]

    syn_lr = torch.tensor(args.lr_teacher).to(args.device)


    ''' training '''
    pde_data_sync = pde_data_sync.detach().to(args.device).requires_grad_(True)
    syn_lr = syn_lr.detach().to(args.device).requires_grad_(True)
    optimizer_pde = torch.optim.Adam([pde_data_sync], lr=args.lr_pde)
    optimizer_lr = torch.optim.SGD([syn_lr], lr=args.lr_lr, momentum=0.5)
    optimizer_pde.zero_grad()

    criterion = nn.MSELoss().to(args.device)
    print('%s training begins'%get_time())

    expert_dir = os.path.join(args.buffer_path, args.filename)
    print("Expert Dir: {}".format(expert_dir))

    if args.load_all:
        buffer = []
        n = 0
        while os.path.exists(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n))):
            buffer = buffer + torch.load(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n)))
            n += 1
        if n == 0:
            raise AssertionError("No buffers detected at {}".format(expert_dir))

    else:
        expert_files = []
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

    best_acc = {m: 0 for m in model_eval_pool}

    best_std = {m: 0 for m in model_eval_pool}

    for it in range(0, args.Iteration+1):
        save_this_it = False

        # writer.add_scalar('Progress', it, it)
        # wandb.log({"Progress": it}, step=it)
        ''' Evaluate synthetic data '''
        if it in eval_it_pool:
            for model_eval in model_eval_pool:
                print('-------------------------\nEvaluation\nmodel_train = %s, model_eval = %s, iteration = %d'%(args.model_name, model_eval, it))

                accs_test = []
                accs_train = []
                for it_eval in range(args.num_eval):
                    net_eval = get_network(args).to(args.device) # get a random model

                    with torch.no_grad():
                        pde_save = pde_data_sync
                    pde_data_sync_eval = copy.deepcopy(pde_save.detach()) # avoid any unaware modification

                    args.lr_net = syn_lr.item()
                    _, acc_train, acc_test = evaluate_synset(it_eval, net_eval, pde_data_sync_eval, syn_grid, test_loader, args)
                    accs_test.append(acc_test)
                    accs_train.append(acc_train)
                accs_test = np.array(accs_test)
                accs_train = np.array(accs_train)
                acc_test_mean = np.mean(accs_test)
                acc_test_std = np.std(accs_test)
                if acc_test_mean > best_acc[model_eval]:
                    best_acc[model_eval] = acc_test_mean
                    best_std[model_eval] = acc_test_std
                    save_this_it = True
                print('Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------'%(len(accs_test), model_eval, acc_test_mean, acc_test_std))
                # wandb.log({'Accuracy/{}'.format(model_eval): acc_test_mean}, step=it)
                # wandb.log({'Max_Accuracy/{}'.format(model_eval): best_acc[model_eval]}, step=it)
                # wandb.log({'Std/{}'.format(model_eval): acc_test_std}, step=it)
                # wandb.log({'Max_Std/{}'.format(model_eval): best_std[model_eval]}, step=it)


        if it in eval_it_pool and (save_this_it or it % 1000 == 0):
            with torch.no_grad():
                pde_save = pde_data_sync.cuda()

                save_dir = os.path.join(".", "logged_files", args.filename)

                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                torch.save(pde_save.cpu(), os.path.join(save_dir, "pdes_{}.pt".format(it)))

                if save_this_it:
                    torch.save(pde_save.cpu(), os.path.join(save_dir, "pdes_best.pt".format(it)))
                # wandb.log({"Pixels": wandb.Histogram(torch.nan_to_num(pde_data_sync.detach().cpu()))}, step=it)

        # wandb.log({"Synthetic_LR": syn_lr.detach().cpu()}, step=it)

        student_net = get_network(args).to(args.device)  # get a random model
        student_net = ReparamModule(student_net)
        if args.distributed:
            student_net = torch.nn.DataParallel(student_net)
        student_net.train()

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

        start_epoch = np.random.randint(0, args.max_start_epoch)
        starting_params = expert_trajectory[start_epoch]

        target_params = expert_trajectory[start_epoch+args.expert_epochs]
        target_params = torch.cat([p.data.to(args.device).reshape(-1) for p in target_params], 0)

        student_params = [torch.cat([p.data.to(args.device).reshape(-1) for p in starting_params], 0).requires_grad_(True)]

        starting_params = torch.cat([p.data.to(args.device).reshape(-1) for p in starting_params], 0)

        syn_pdes = pde_data_sync

        y_hat = label_syn.to(args.device)

        param_loss_list = []
        param_dist_list = []
        indices_chunks = []

        if args.batch_syn is None:
            args.batch_syn = args.sync_num

        for step in range(args.syn_steps):

            if not indices_chunks:
                indices = torch.randperm(len(syn_pdes))
                indices_chunks = list(torch.split(indices, args.batch_syn))

            these_indices = indices_chunks.pop()


            x = syn_pdes[these_indices][..., :-1]
            this_y = syn_pdes[these_indices][..., -1]

            if args.distributed:
                forward_params = student_params[-1].unsqueeze(0).expand(torch.cuda.device_count(), -1)
            else:
                forward_params = student_params[-1]
          
            x = student_net(x, syn_grid, flat_param=forward_params)
            ce_loss = criterion(x, this_y)

            grad = torch.autograd.grad(ce_loss, student_params[-1], create_graph=True)[0]

            student_params.append(student_params[-1] - syn_lr * grad)


        param_loss = torch.tensor(0.0).to(args.device)
        param_dist = torch.tensor(0.0).to(args.device)

        param_loss += torch.nn.functional.mse_loss(student_params[-1], target_params, reduction="sum")
        param_dist += torch.nn.functional.mse_loss(starting_params, target_params, reduction="sum")

        param_loss_list.append(param_loss)
        param_dist_list.append(param_dist)


        param_loss /= num_params
        param_dist /= num_params

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

        if it%10 == 0:
            print('%s iter = %04d, loss = %.4f' % (get_time(), it, grand_loss.item()))

    # wandb.finish()


if __name__ == '__main__':
    main()
