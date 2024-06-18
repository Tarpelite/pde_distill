import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import h5py
import torch.utils
from torch.utils.data import Dataset
import math as mt
import fno
from tqdm import tqdm


class FNODatasetSingle(Dataset):
    def __init__(self, filename,
                 initial_step=10,
                 saved_folder='../data/',
                 reduced_resolution=1,
                 reduced_resolution_t=1,
                 reduced_batch=1,
                 if_test=False,
                 test_ratio=0.1,
                 num_samples_max = -1
                 ):
        """
        
        :param filename: filename that contains the dataset
        :type filename: STR
        :param filenum: array containing indices of filename included in the dataset
        :type filenum: ARRAY
        :param initial_step: time steps taken as initial condition, defaults to 10
        :type initial_step: INT, optional

        """
        
        # Define path to files
        root_path = os.path.join(os.path.abspath(saved_folder), filename)
        if filename[-2:] != 'h5':
            print(f".HDF5 file extension is assumed hereafter")
        
            with h5py.File(root_path, 'r', swmr=True) as f:
                keys = list(f.keys())
                keys.sort()
                if 'tensor' not in keys:
                    _data = np.array(f['density'], dtype=np.float32)  # batch, time, x,...
                    idx_cfd = _data.shape
                    if len(idx_cfd)==3:  # 1D
                        self.data = np.zeros([idx_cfd[0]//reduced_batch,
                                              idx_cfd[2]//reduced_resolution,
                                              mt.ceil(idx_cfd[1]/reduced_resolution_t),
                                              3],
                                            dtype=np.float32)
                        #density
                        _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution]
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data[:, :, :], (0, 2, 1))
                        self.data[...,0] = _data   # batch, x, t, ch
                        # pressure
                        _data = np.array(f['pressure'], dtype=np.float32)  # batch, time, x,...
                        _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution]
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data[:, :, :], (0, 2, 1))
                        self.data[...,1] = _data   # batch, x, t, ch
                        # Vx
                        _data = np.array(f['Vx'], dtype=np.float32)  # batch, time, x,...
                        _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution]
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data[:, :, :], (0, 2, 1))
                        self.data[...,2] = _data   # batch, x, t, ch

                        self.grid = np.array(f["x-coordinate"], dtype=np.float32)
                        self.grid = torch.tensor(self.grid[::reduced_resolution], dtype=torch.float).unsqueeze(-1)
                        print(self.data.shape)
                    if len(idx_cfd)==4:  # 2D
                        self.data = np.zeros([idx_cfd[0]//reduced_batch,
                                              idx_cfd[2]//reduced_resolution,
                                              idx_cfd[3]//reduced_resolution,
                                              mt.ceil(idx_cfd[1]/reduced_resolution_t),
                                              4],
                                             dtype=np.float32)
                        # density
                        _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution,::reduced_resolution]
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data, (0, 2, 3, 1))
                        self.data[...,0] = _data   # batch, x, t, ch
                        # pressure
                        _data = np.array(f['pressure'], dtype=np.float32)  # batch, time, x,...
                        _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution,::reduced_resolution]
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data, (0, 2, 3, 1))
                        self.data[...,1] = _data   # batch, x, t, ch
                        # Vx
                        _data = np.array(f['Vx'], dtype=np.float32)  # batch, time, x,...
                        _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution,::reduced_resolution]
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data, (0, 2, 3, 1))
                        self.data[...,2] = _data   # batch, x, t, ch
                        # Vy
                        _data = np.array(f['Vy'], dtype=np.float32)  # batch, time, x,...
                        _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution,::reduced_resolution]
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data, (0, 2, 3, 1))
                        self.data[...,3] = _data   # batch, x, t, ch

                        x = np.array(f["x-coordinate"], dtype=np.float32)
                        y = np.array(f["y-coordinate"], dtype=np.float32)
                        x = torch.tensor(x, dtype=torch.float)
                        y = torch.tensor(y, dtype=torch.float)
                        X, Y = torch.meshgrid(x, y, indexing='ij')
                        self.grid = torch.stack((X, Y), axis=-1)[::reduced_resolution, ::reduced_resolution]
                
                    if len(idx_cfd)==5:  # 3D
                        self.data = np.zeros([idx_cfd[0]//reduced_batch,
                                              idx_cfd[2]//reduced_resolution,
                                              idx_cfd[3]//reduced_resolution,
                                              idx_cfd[4]//reduced_resolution,
                                              mt.ceil(idx_cfd[1]/reduced_resolution_t),
                                              5],
                                             dtype=np.float32)
                        # density
                        _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution,::reduced_resolution,::reduced_resolution]
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data, (0, 2, 3, 4, 1))
                        self.data[...,0] = _data   # batch, x, t, ch
                        # pressure
                        _data = np.array(f['pressure'], dtype=np.float32)  # batch, time, x,...
                        _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution,::reduced_resolution,::reduced_resolution]
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data, (0, 2, 3, 4, 1))
                        self.data[...,1] = _data   # batch, x, t, ch
                        # Vx
                        _data = np.array(f['Vx'], dtype=np.float32)  # batch, time, x,...
                        _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution,::reduced_resolution,::reduced_resolution]
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data, (0, 2, 3, 4, 1))
                        self.data[...,2] = _data   # batch, x, t, ch
                        # Vy
                        _data = np.array(f['Vy'], dtype=np.float32)  # batch, time, x,...
                        _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution,::reduced_resolution,::reduced_resolution]
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data, (0, 2, 3, 4, 1))
                        self.data[...,3] = _data   # batch, x, t, ch
                        # Vz
                        _data = np.array(f['Vz'], dtype=np.float32)  # batch, time, x,...
                        _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution,::reduced_resolution,::reduced_resolution]
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data, (0, 2, 3, 4, 1))
                        self.data[...,4] = _data   # batch, x, t, ch

                        x = np.array(f["x-coordinate"], dtype=np.float32)
                        y = np.array(f["y-coordinate"], dtype=np.float32)
                        z = np.array(f["z-coordinate"], dtype=np.float32)
                        x = torch.tensor(x, dtype=torch.float)
                        y = torch.tensor(y, dtype=torch.float)
                        z = torch.tensor(z, dtype=torch.float)
                        X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
                        self.grid = torch.stack((X, Y, Z), axis=-1)[::reduced_resolution,\
                                                                    ::reduced_resolution,\
                                                                    ::reduced_resolution]
                                                                    
                else:  # scalar equations
                    ## data dim = [t, x1, ..., xd, v]
                    _data = np.array(f['tensor'], dtype=np.float32)  # batch, time, x,...
                    if len(_data.shape) == 3:  # 1D
                        _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution]
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data[:, :, :], (0, 2, 1))
                        self.data = _data[:, :, :, None]  # batch, x, t, ch

                        self.grid = np.array(f["x-coordinate"], dtype=np.float32)
                        self.grid = torch.tensor(self.grid[::reduced_resolution], dtype=torch.float).unsqueeze(-1)
                    if len(_data.shape) == 4:  # 2D Darcy flow
                        # u: label
                        _data = _data[::reduced_batch,:,::reduced_resolution,::reduced_resolution]
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data[:, :, :, :], (0, 2, 3, 1))
                        #if _data.shape[-1]==1:  # if nt==1
                        #    _data = np.tile(_data, (1, 1, 1, 2))
                        self.data = _data
                        # nu: input
                        _data = np.array(f['nu'], dtype=np.float32)  # batch, time, x,...
                        _data = _data[::reduced_batch, None,::reduced_resolution,::reduced_resolution]
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data[:, :, :, :], (0, 2, 3, 1))
                        self.data = np.concatenate([_data, self.data], axis=-1)
                        self.data = self.data[:, :, :, :, None]  # batch, x, y, t, ch

                        x = np.array(f["x-coordinate"], dtype=np.float32)
                        y = np.array(f["y-coordinate"], dtype=np.float32)
                        x = torch.tensor(x, dtype=torch.float)
                        y = torch.tensor(y, dtype=torch.float)
                        X, Y = torch.meshgrid(x, y, indexing='ij')
                        self.grid = torch.stack((X, Y), axis=-1)[::reduced_resolution, ::reduced_resolution]

        elif filename[-2:] == 'h5':  # SWE-2D (RDB)
            print(f".H5 file extension is assumed hereafter")
        
            with h5py.File(root_path, 'r') as f:
                keys = list(f.keys())
                keys.sort()
                data_arrays = [np.array(f[key]['data'], dtype=np.float32) for key in keys]
                _data = torch.from_numpy(np.stack(data_arrays, axis=0))   # [batch, nt, nx, ny, nc]
                _data = _data[::reduced_batch, ::reduced_resolution_t, ::reduced_resolution, ::reduced_resolution, ...]
                _data = torch.permute(_data, (0, 2, 3, 1, 4))   # [batch, nx, ny, nt, nc]
                gridx, gridy = np.array(f['0023']['grid']['x'], dtype=np.float32), np.array(f['0023']['grid']['y'], dtype=np.float32)
                mgridX, mgridY = np.meshgrid(gridx, gridy, indexing='ij')
                _grid = torch.stack((torch.from_numpy(mgridX), torch.from_numpy(mgridY)), axis=-1)
                grid = _grid[::reduced_resolution, ::reduced_resolution, ...]
                _tsteps_t = torch.from_numpy(np.array(f['0023']['grid']['t'], dtype=np.float32))
                tsteps_t = _tsteps_t[::reduced_resolution_t]
                self.data = _data
                self.grid = _grid
                self.tsteps_t = tsteps_t

        if num_samples_max>0:
            num_samples_max  = min(num_samples_max, self.data.shape[0])
        else:
            num_samples_max = self.data.shape[0]

        test_idx = int(num_samples_max * test_ratio)
        if if_test:
            self.data = self.data[:test_idx]
        else:
            self.data = self.data[test_idx:num_samples_max]

        # Time steps used as initial conditions
        self.initial_step = initial_step

        self.data = self.data if torch.is_tensor(self.data) else torch.tensor(self.data)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx,...,:self.initial_step,:], self.data[idx], self.grid


def get_dataset(data_dir, filename, batch_size=1, num_workers=2, args=None):
    
    train_data = FNODatasetSingle(filename,
                                reduced_resolution=args.reduced_resolution,
                                reduced_resolution_t=args.reduced_resolution_t,
                                reduced_batch=args.reduced_batch,
                                initial_step=args.initial_step,
                                saved_folder = data_dir
                                )

    val_data = FNODatasetSingle(filename,
                                reduced_resolution=args.reduced_resolution,
                                reduced_resolution_t=args.reduced_resolution_t,
                                reduced_batch=args.reduced_batch,
                                initial_step=args.initial_step,
                                saved_folder = data_dir,
                                if_test=True
                                )
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    return train_loader, test_loader


def get_network(args):

    if args.model_name == "FNO1D":
        net = fno.FNO1d(num_channels=args.num_channel, modes=args.modes, initial_step=args.initial_step, width=args.width)
    elif args.model_name == "FNO2D":
        net = fno.FNO2d(num_channels=args.num_channel, modes=args.modes, initial_step=args.initial_step, width=args.width)
    elif args.model_name == "FNO3D":
        net = fno.FNO3d(num_channels=args.num_channel, modes=args.modes, initial_step=args.initial_step, width=args.width)

    gpu_num = torch.cuda.device_count()
    
    if gpu_num > 0:
        device = "cuda"
    else:
        device = "cpu"

    net = net.to(device)
    return net

def get_time():
    return str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))


def epoch(mode, dataloader, net, optimizer, criterion, args):
    loss, loss_avg, num_exp = 0, 0, 0
    net = net.to(args.device)


    if mode == 'train':
        net.train()
    else:
        net.eval()

    for i_batch in tqdm(dataloader):
        xx, yy, grid = i_batch

        # xx: input tensor (first few time steps) [b, x1, ..., xd, t_init, v]
        # yy: target tensor [b, x1, ..., xd, t, v]
        # grid: meshgrid [b, x1, ..., xd, dims]
        device = torch.device(args.device)
        xx = xx.to(device)
        yy = yy.to(device)
        grid = grid.to(device)

        # Initialize the prediction tensor
        pred = yy[..., :args.initial_step, :]
        # Extract shape of the input tensor for reshaping (i.e. stacking the
        # time and channels dimension together)
        inp_shape = list(xx.shape)
        inp_shape = inp_shape[:-2]
        inp_shape.append(-1)

        # train in single style
        x = xx[..., :args.t_train-1 , :]
        y = yy[..., args.t_train-1:args.t_train, :]

        # import pudb;pu.db;
        pred = net(x.reshape(x.shape[0], x.shape[1],x.shape[2]*x.shape[-1]), grid)
        _batch = yy.size(0)
        # import pudb;pu.db;
        loss_avg = criterion(pred.reshape(_batch, -1), y.reshape(_batch, -1))

        if mode == 'train':
            optimizer.zero_grad()
            loss_avg.backward()
            optimizer.step()

        n_b = xx.shape[0]

        loss += loss_avg.item() * n_b
        # acc_avg += acc
        num_exp += n_b

    loss /= num_exp
    # acc_avg /= num_exp

    return loss


class TensorDataset(Dataset):
    def __init__(self, pde_data, grid, args):
        self.pde_data = pde_data.detach().float()
        self.grid = grid[0]
        self.initial_step = args.initial_step
    
    def __len__(self):
        return self.pde_data.shape[0]
    
    def __getitem__(self, index):
        return self.pde_data[index, ..., :self.initial_step, :], self.pde_data[index], self.grid 
        



def evaluate_synset(it_eval, net, pde_data, grid, testloader, args):
    
    device = torch.device(args.device)
    net = net.to(device)
    pde_data  = pde_data.to(device)
    grid = grid.to(device)

    lr = float(args.lr_net)
    Epoch = int(args.epoch_eval_train)
    lr_schedule = [Epoch//2+1]
    optimizer = torch.optim.Adam(net.parameters(), lr=lr,  weight_decay=0.0005)

    criterion = nn.MSELoss().to(device)
    dst_train = TensorDataset(pde_data, grid, args)
    train_loader = torch.utils.data.DataLoader(dst_train,batch_size=args.batch_syn_eval, shuffle=True, num_workers=0)


    start = time.time()
    loss_train_list = []

    for ep in tqdm(range(Epoch+1)):
        loss_train = epoch('train', dataloader=train_loader, net=net, optimizer=optimizer, criterion=criterion, args=args)
        loss_train_list.append(loss_train)
        if ep == Epoch:
            with torch.no_grad():
                loss_test = epoch('test', testloader, net, optimizer, criterion, args, )
        if ep in lr_schedule:
            lr *= 0.1
            optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=0.0005)


    time_train = time.time() - start

    print('%s Evaluate_%02d: epoch = %04d train time = %d s train mse = %.6f , test mse = %.6f' % (get_time(), it_eval, Epoch, int(time_train), loss_train, loss_test))

    return net, loss_train_list, loss_test

