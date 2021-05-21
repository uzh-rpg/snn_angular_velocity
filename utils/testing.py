import numpy as np
import os
import torch
from tqdm import tqdm

from data_loader.testing import TestDatabase, TrainDatabase
from model.loss import compute_loss
from model.metric import medianRelativeError, rmse
from model.cnn5_avgp_fc1 import SNN
from .gpu import moveToGPUDevice
from .tbase import TBase


class Tester(TBase):
    def __init__(self, data_dir, write, log_config, general_config):
        super().__init__(data_dir, log_config, general_config)
        self.write_output = write
        self.output_dir = self.log_config.getOutDir()

        test_database = TestDatabase(self.data_dir)
        self.test_loader = torch.utils.data.DataLoader(
                test_database,
                batch_size=general_config['batchsize'],
                shuffle=False,
                num_workers=general_config['hardware']['readerThreads'],
                pin_memory=True,
                drop_last=False)

        self.data_collector = DataCollector(general_config['simulation']['tStartLoss'])

    def test(self):
        self._loadNetFromCheckpoint()
        self.net = self.net.eval()
        #print(self.net)
        with torch.no_grad():
            for data in tqdm(self.test_loader, desc='testing'):
                data = moveToGPUDevice(data, self.device, self.dtype)

                spike_tensor = data['spike_tensor']
                ang_vel_gt = data['angular_velocity']

                ang_vel_pred = self.net(spike_tensor)
                self.data_collector.append(ang_vel_pred, ang_vel_gt, data['file_number'])
        if self.write_output:
            self.data_collector.writeToDisk(self.output_dir)
        self.data_collector.printErrors()

class Trainer(TBase):
    def __init__(self, data_dir, write, log_config, general_config):
        super().__init__(data_dir, log_config, general_config)
        self.write_output = write
        self.output_dir = self.log_config.getOutDir()
        test_database = TestDatabase(self.data_dir)
        train_database = TrainDatabase(self.data_dir)
        
        self.train_loader = torch.utils.data.DataLoader(
                train_database,
                batch_size=general_config['batchsize'],
                shuffle=True,
                num_workers=general_config['hardware']['readerThreads'],
                pin_memory=True,
                drop_last=False)

        self.test_loader = torch.utils.data.DataLoader(
                test_database,
                batch_size=general_config['batchsize'],
                shuffle=False,
                num_workers=general_config['hardware']['readerThreads'],
                pin_memory=True,
                drop_last=False)
        self.data_collector = DataCollector(general_config['simulation']['tStartLoss'])

    def train(self, num_epochs, learning_rate):
        self._trainfromscratch()    # train from scratch
        #self._loadNetFromCheckpoint()      # train from pretrained model
        self.net = self.net.train()
        print(self.net)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate)
        for epoch in range(num_epochs):
            for i , data in enumerate(self.train_loader):
            #for data in tqdm(self.train_loader, desc='training'):
                data = moveToGPUDevice(data, self.device, self.dtype)

                spike_tensor = data['spike_tensor']
                ang_vel_gt = data['angular_velocity']

                ang_vel_pred = self.net(spike_tensor)
                loss = compute_loss(ang_vel_pred, ang_vel_gt, 50)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                self.data_collector.append(ang_vel_pred, ang_vel_gt, data['file_number'])
                if (i+1) % 100 == 0:
                    print('Epoch: [{}/{}], Step: [{}/{}], Loss: {}'
                        .format(epoch+1, num_epochs, i+1, len(self.train_loader), loss.item()))
        if self.write_output:
            self.data_collector.writeToDisk(self.output_dir)
        self.data_collector.printErrors()

    def test(self):
        self.net = self.net.eval()

        with torch.no_grad():
            for data in tqdm(self.test_loader, desc='testing'):
                data = moveToGPUDevice(data, self.device, self.dtype)

                spike_tensor = data['spike_tensor']
                ang_vel_gt = data['angular_velocity']

                ang_vel_pred = self.net(spike_tensor)
                self.data_collector.append(ang_vel_pred, ang_vel_gt, data['file_number'])
        if self.write_output:
            self.data_collector.writeToDisk(self.output_dir)
        self.data_collector.printErrors()



class DataCollector:
    def __init__(self, loss_start_idx: int):
        assert loss_start_idx >= 0
        self.loss_start_idx = loss_start_idx

        self.file_indices = list()
        self.data_gt = list()
        self.data_pred = list()

    def append(self, pred: torch.Tensor, gt: torch.Tensor, file_indices: torch.Tensor):
        # pred/gt: (batchsize, 3, time) tensor
        # file_indices: (batchsize) tensor

        self.data_pred.append(pred.detach().cpu().numpy())
        self.data_gt.append(gt.detach().cpu().numpy())
        self.file_indices.append(file_indices.to(torch.long).cpu().numpy())

    def writeToDisk(self, out_dir: str):
        pred = np.concatenate(self.data_pred, axis=0)
        gt = np.concatenate(self.data_gt, axis=0)
        file_indices = np.concatenate(self.file_indices)
        np.save(os.path.join(out_dir, 'predictions.npy'), pred)
        np.save(os.path.join(out_dir, 'groundtruth.npy'), gt)
        np.save(os.path.join(out_dir, 'indices.npy'), file_indices)

    def printErrors(self):
        pred = np.concatenate(self.data_pred, axis=0)
        gt = np.concatenate(self.data_gt, axis=0)

        pred = pred[..., self.loss_start_idx:]
        gt = gt[..., self.loss_start_idx:]

        print('RMSE: {} deg/s'.format(rmse(pred, gt, deg=True)))
        print('median of relative error: {}'.format(medianRelativeError(pred, gt)))
