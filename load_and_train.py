
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm 
import torch.nn.functional as F
class Deep_sfmx(nn.Module):
    def __init__(self):
        super(Deep_sfmx, self).__init__()
        self.conv1    = nn.Conv1d(30, 64,kernel_size=7, stride=3)
        self.conv2    = nn.Conv1d(64, 128,kernel_size=5, stride=3)
        self.bnorm    = nn.BatchNorm1d(64)
        self.pool2 =  nn.AdaptiveAvgPool1d(output_size=1)
        self.pooling  = nn.MaxPool2d((1))
        self.flatten  = nn.Flatten()
        self.fcl_1    = nn.Linear(128, 64)
        self.fcl_2 = nn.Linear(64,9)
        self.fc1_3 = nn.Linear(9,1)
        self.sfmx     = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(.2) 
    def forward(self, x):
        x = x.transpose(-1, -2)
        x = x.cuda()
        x = self.pooling(F.relu(self.conv1(x)))
        # print(x.shape)

        x = self.pool2(F.relu(self.conv2(x)))
        x = self.flatten(x)
        x = F.relu(self.fcl_1(x))
        x = self.dropout(x)
        x = self.fcl_2(x)
        
        x = self.sfmx(x)
        return x
 

 

out_fname = 'F:\\Dome_Pilot\\Synced\\'
import os
pt_file = torch.load(os.path.join(out_fname,"train_data.pt"))
from torch.nn.utils.rnn import pad_sequence
def pad_collate(batch):
  (xx, yy) = zip(*batch)
  x_lens = [len(x) for x in xx]
  y_lens = [len(y) for y in yy]

  xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)
  yy_pad = pad_sequence(yy, batch_first=True, padding_value=0)

  return xx_pad, yy_pad, x_lens, y_lens



#train_loader = DataLoader(list(zip(pt_file['imu'], torch.nn.functional.one_hot(pt_file['label'].long(),num_classes=9))), shuffle=True,collate_fn=pad_collate, batch_size=48)
train_loader = DataLoader(list(zip(pt_file['imu'],torch.nn.functional.one_hot(pt_file['label'].long()))), shuffle=True, batch_size=32)
test_file = torch.load(out_fname+"test_data.pt")

#test_loader = DataLoader(list(zip(test_file['imu'], torch.nn.functional.one_hot(pt_file['label'].long(),num_classes=9))), shuffle=True,collate_fn=pad_collate, batch_size=48)

test_loader = DataLoader(list(zip(test_file['imu'], torch.nn.functional.one_hot(pt_file['label'].long()))), shuffle=True, batch_size=32)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = Deep_sfmx()
# model = FC(1, 1, 1)
model.to(device)

# TODO: define Cross Entropy Loss 
# out = model()
lossFunction = nn.CrossEntropyLoss()

# TODO: create Adam Optimizer and define your hyperparameters 
learning_rate = .001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,betas=(0.9, 0.999))

# Train the model
num_epochs = 100


count = 0
loss_list = []
iteration_list = []
accuracy_list = []
#import pdb; pdb.set_trace()
for epoch in tqdm(range(num_epochs)):
    model.train()
    print(epoch)

    for i, (imu, labels) in enumerate(train_loader):
        
        imu, labels = imu.to(device), labels.to(device)


        
        # Clear gradients
        optimizer.zero_grad()
        
        # TODO: Forward propagation
        outputs = model(imu)
    
        # TODO: Calculate softmax and ross entropy loss
        
        loss = lossFunction(outputs.squeeze(1), torch.argmax(labels.squeeze(1),1))#labels.float().squeeze(1))
        # Backprop agate your Loss 
        loss.backward()
        
        # Update CNN model  
        optimizer.step()
        
        count += 1
        
        if count % 10 == 0:
            model.eval()
            # Calculate Accuracy         
            correct = 0
            total = 0
            # Iterate through test dataset
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                
                # Forward propagation
                outputs = model(images)
                
                # Get predictions from the maximum value
                predicted = torch.argmax(outputs,1)
                
                true_labels = torch.argmax(labels.squeeze(1),1)

                # Total number of labels
                total += len(true_labels)

                correct += (predicted == true_labels).sum()


            
            accuracy = 100 * correct / float(total)
            
            # store loss and iteration
            loss_list.append(loss.data.cpu())
            iteration_list.append(count)
            accuracy_list.append(accuracy.cpu())

            if accuracy >= 70:
              break
        if count % 10 == 0:
            #import pdb; pdb.set_trace()
            # Print Loss
            print('Iteration: {}  Loss: {}  Accuracy: {} %'.format(count, loss.data, accuracy))

import matplotlib.pyplot as plt
plt.figure()
plt.plot(loss_list)

plt.figure()
plt.plot(accuracy_list)
plt.show()