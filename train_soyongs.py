import os
import time
import argparse
import os.path as osp
from glob import glob

import torch
import joblib
import numpy as np
from torch import nn
from tqdm import tqdm

import constants as _C
from dataset_soyongs import Dataset, make_collate_fn
from balanced_loss import Loss

class Network(nn.Module):
    def __init__(self, num_classes=8):
        super(Network, self).__init__()
        self.conv1      = nn.Conv1d(30, 64,kernel_size=7, stride=3)
        self.bn1        = nn.BatchNorm1d(64)
        self.conv2      = nn.Conv1d(64, 128,kernel_size=5, stride=3)
        self.bn2        = nn.BatchNorm1d(128)
        self.pool      = nn.AdaptiveAvgPool1d(output_size=1)
        self.fc1        = nn.Linear(128, 64)
        self.fc2        = nn.Linear(64, num_classes)
        self.sfmx       = nn.Softmax(dim=1)
        self.dropout    = nn.Dropout(.2) 
        self.relu       = nn.ReLU()

    def forward(self, x):
        b, f, d = x.shape[:3]
        
        x = x.transpose(-1, -2) # batch frame dim -> batch dim frame
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = self.pool(x)
        
        x = x.squeeze(dim=-1)
        x = self.dropout(self.fc1(x))
        x = self.fc2(x)
        x = self.sfmx(x)
        
        return x
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr-decay', type=float, default=1e-1)
    parser.add_argument('--lr-patience', type=int, default=5)
    parser.add_argument('--use-gpu', action='store_true')
    args = parser.parse_args()
    
    device = 'cuda' if args.use_gpu and torch.cuda.is_available() else 'cpu'
            
    train_dloader = torch.utils.data.DataLoader(Dataset(True), 
                                                batch_size=args.batch_size, 
                                                num_workers=args.num_workers, 
                                                shuffle=True, 
                                                pin_memory=True,
                                                collate_fn=make_collate_fn())
    
    test_dloader = torch.utils.data.DataLoader(Dataset(False), 
                                               batch_size=1, 
                                               num_workers=0, 
                                               shuffle=False, 
                                               pin_memory=True,
                                               collate_fn=make_collate_fn())
    
    net = Network().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999))
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=args.lr_decay,
        patience=args.lr_patience,
        verbose=False,
    )
    
    ce_loss = Loss(
        loss_type="cross_entropy",
            samples_per_class=train_dloader.dataset.counts,
            class_balanced=True
        )
    
    for epoch in (ebar := tqdm(range(1, args.epoch + 1))):
        ebar.set_description_str(f'Epoch [{epoch}|{args.epoch}]')
        
        # ======== Train one epoch ======== #
        for i, batch in (ibar := tqdm(enumerate(train_dloader), total=len(train_dloader), leave=False)):
            imu = batch['imu'].to(device)
            label = batch['label'].to(device)
            
            pred = net(imu)
            loss = ce_loss(pred, label)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            ibar.set_description_str(f'Loss: {loss.item():.3f}')
        
        # ======== Evaluate one epoch ======== #
        net.eval()
        accs = []
        for batch in test_dloader:
            with torch.no_grad():
                imu = batch['imu'].to(device).squeeze(0)
                label = batch['label'].squeeze(0)
                
                pred = net(imu)
                pred = torch.argmax(pred, dim=-1)
                accs.append((pred == label).cpu().numpy())
        
        accs = np.concatenate(accs)
        acc = accs.astype(float).mean() * 1e2
        lr_scheduler.step(acc)
        ebar.set_postfix_str(f"Accuracy: {acc:.2f} % ")
        net.train()
