import os
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm, trange
from utils import get_dataset, get_network, get_daparam,\
    TensorDataset, epoch
import copy

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def main(args):

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'


    train_loader,test_loader = get_dataset(args.dataset, args.data_path, args.batch_real,  args=args)

    # print('\n================== Exp %d ==================\n '%exp)
    print('Hyper-parameters: \n', args.__dict__)

    save_dir = os.path.join(args.buffer_path, args.dataset)
   
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


    ''' organize the real dataset '''
    # images_all = []
    # labels_all = []
    # indices_class = [[] for c in range(num_classes)]
    # print("BUILDING DATASET")
    # for i in trange(len(dst_train)):
    #     sample = dst_train[i]
    #     images_all.append(torch.unsqueeze(sample[0], dim=0))
    #     labels_all.append(class_map[torch.tensor(sample[1]).item()])

    # for i, lab in tqdm(enumerate(labels_all)):
    #     indices_class[lab].append(i)
    # images_all = torch.cat(images_all, dim=0).to("cpu")
    # labels_all = torch.tensor(labels_all, dtype=torch.long, device="cpu")

    # for c in range(num_classes):
    #     print('class c = %d: %d real images'%(c, len(indices_class[c])))

    # for ch in range(channel):
    #     print('real images channel %d, mean = %.4f, std = %.4f'%(ch, torch.mean(images_all[:, ch]), torch.std(images_all[:, ch])))

    criterion = nn.MSELoss().to(args.device)

    trajectories = []


    for it in range(0, args.num_experts):

        ''' Train synthetic data '''
        teacher_net = get_network(args.model, args.num_channel, args).to(args.device) # get a random model
        teacher_net.train()
        lr = args.lr_teacher
        teacher_optim = torch.optim.Adam(teacher_net.parameters(), lr=lr, momentum=args.mom, weight_decay=args.l2)  # optimizer_img for synthetic data
        teacher_optim.zero_grad()

        timestamps = []

        timestamps.append([p.detach().cpu() for p in teacher_net.parameters()])

        lr_schedule = [args.train_epochs // 2 + 1]

        for e in range(args.train_epochs):

            train_loss, train_acc = epoch("train", dataloader=train_loader, net=teacher_net, optimizer=teacher_optim,
                                        criterion=criterion, args=args, aug=True)

            test_loss, test_acc = epoch("test", dataloader=test_loader, net=teacher_net, optimizer=None,
                                        criterion=criterion, args=args, aug=False)

            print("Itr: {}\tEpoch: {}\tTrain Acc: {}\tTest Acc: {}".format(it, e, train_acc, test_acc))

            timestamps.append([p.detach().cpu() for p in teacher_net.parameters()])

            if e in lr_schedule and args.decay:
                lr *= 0.1
                teacher_optim = torch.optim.Adam(teacher_net.parameters(), lr=lr, momentum=args.mom, weight_decay=args.l2)
                teacher_optim.zero_grad()

        trajectories.append(timestamps)

        if len(trajectories) == args.save_interval:
            n = 0
            while os.path.exists(os.path.join(save_dir, "replay_buffer_{}.pt".format(n))):
                n += 1
            print("Saving {}".format(os.path.join(save_dir, "replay_buffer_{}.pt".format(n))))
            torch.save(trajectories, os.path.join(save_dir, "replay_buffer_{}.pt".format(n)))
            trajectories = []


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--model_name', type=str, default='FNO1D', help='model')
    parser.add_argument('--num_experts', type=int, default=100, help='training iterations')
    parser.add_argument("--reduced_resolution", type=int, default=8)
    parser.add_argument("--reduced_resolution_t", type=int, default=5)
    parser.add_argument("--t_train", type=int,default=100)
    parser.add_argument("--modes", type=int, default=16)
    parser.add_argument("--width", default=20, type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument('--lr_teacher', type=float, default=0.01, help='learning rate for updating network parameters')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real loader')
    parser.add_argument('--num_channel', type=int, default=3, help='batch size for real loader')
    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument('--train_epochs', type=int, default=50)
    parser.add_argument('--mom', type=float, default=0, help='momentum')
    parser.add_argument('--l2', type=float, default=0, help='l2 regularization')
    parser.add_argument('--save_interval', type=int, default=10)

    args = parser.parse_args()
    main(args)

