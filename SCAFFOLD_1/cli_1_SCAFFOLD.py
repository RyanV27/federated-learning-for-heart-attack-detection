import torch
import torch.nn as nn
import json
import pickle

class ConvNetQuake(nn.Module):
    def __init__(self, B, E, lr, name):
        super(ConvNetQuake, self).__init__()

        self.name = name
        self.B = B
        self.E = E
        self.len = 0
        self.lr = lr
        self.loss = 0
        self.control = {}
        self.delta_control = {}
        self.delta_y = {}

        self.conv1 = nn.Conv1d(in_channels=2, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.conv5 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.conv6 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.conv7 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.conv8 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.linear1 = nn.Linear(1280, 128)
        self.linear2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()
        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(32)
        self.bn3 = nn.BatchNorm1d(32)
        self.bn4 = nn.BatchNorm1d(32)
        self.bn5 = nn.BatchNorm1d(32)
        self.bn6 = nn.BatchNorm1d(32)
        self.bn7 = nn.BatchNorm1d(32)
        self.bn8 = nn.BatchNorm1d(32)

        for k, v in self.named_parameters():
            self.control[k] = torch.zeros_like(v.data)
            self.delta_control[k] = torch.zeros_like(v.data)
            self.delta_y[k] = torch.zeros_like(v.data)

    def forward(self, x):
        x = self.bn1(F.relu((self.conv1(x))))
        x = self.bn2(F.relu((self.conv2(x))))
        x = self.bn3(F.relu((self.conv3(x))))
        x = self.bn4(F.relu((self.conv4(x))))
        x = self.bn5(F.relu((self.conv5(x))))
        x = self.bn6(F.relu((self.conv6(x))))
        x = self.bn7(F.relu((self.conv7(x))))
        x = self.bn8(F.relu((self.conv8(x))))
        x = torch.reshape(x, (10, -1))
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.sigmoid(x)

        return x
    
K, C, E, B, r = 10, 0.5, 30, 50, 10
input_dim = 32
lr = 1.0e-4
model = ConvNetQuake(B, E, lr, name = 'cli_1')

#for k, v in model.named_parameters():
#    print(k)

#for key in model.state_dict().keys():
#    print(key)

file_path = 'dictionary.pkl'

with open(file_path, 'wb') as f:
    pickle.dump(model.control, f)

with open(file_path, 'rb') as f:
    my_dict = pickle.load(f)

model.control = my_dict
print(model.control)