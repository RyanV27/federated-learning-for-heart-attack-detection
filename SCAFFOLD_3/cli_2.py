from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import wfdb
import time
import random
from sklearn.preprocessing import minmax_scale
import sys
from torch.utils.tensorboard import SummaryWriter
import socket
import pickle
import copy
from torch.optim.lr_scheduler import StepLR
from ScaffoldOptimizer import ScaffoldOptimizer

channel_1 = 'v6'
channel_2 = 'vz'
seed_num = 30
run_num = 1
folder_name = 'ptbdb_data_cli2/'
client_no = 2
fed_iteration_no = 5
window_size = 10000

# Define your neural network model
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

#############################################################

# Returns a list of records common to both lst1 and lst2
def intersection(lst1, lst2): 
    return list(set(lst1) & set(lst2))

##############################################################

def toReducedArray(original_data):
    data = original_data
    min_arr_size = data[0][0].shape[0]
    
    for i in range(len(data)):
        # for j in range(len(data[i])):       # For two channels
        for j in range(len(data[i]) - 1):
            if(data[i][j].shape[0] < min_arr_size):
                min_arr_size = data[i][j].shape[0]

    # print("Minimum array size = ", min_arr_size)

    data_reduced = np.zeros(( len(data), len(data[0]), min_arr_size ))
    #print("Shape of data_healthy_train_reduced: ", data_reduced.shape)
    
    for i in range(len(data)):
        # for j in range(len(data[i])):      # For two channels
        for j in range(len(data[i]) - 1):  
            data_reduced[i][j] = data[i][j][:min_arr_size]
            
    #print("Reduced data array:-/n", data_reduced)
    return(data_reduced)

#############################################################

def load_data():
    with open(folder_name + 'RECORDS') as fp:  
        lines = fp.readlines()
#    print(lines)
    
    # To find out files with healthy and unhealthy patients(suffering from myocardial infarction)
    files_unhealthy, files_healthy = [], []

    for file in lines:
        file_path = folder_name + file[:-1] + ".hea"
        
        # read header to determine class
        if 'Myocardial infarction' in open(file_path).read():
            files_unhealthy.append(file)
            
        if 'Healthy control' in open(file_path).read():
            files_healthy.append(file)

    # shuffle data (cross-validation)
    np.random.seed(int(seed_num))
    np.random.shuffle(files_unhealthy)
    np.random.shuffle(files_healthy)

    healthy_train = files_healthy[:int(0.8*len(files_healthy))]
    healthy_val = files_healthy[int(0.8*len(files_healthy)):]
    unhealthy_train = files_unhealthy[:int(0.8*len(files_unhealthy))]
    unhealthy_val = files_unhealthy[int(0.8*len(files_unhealthy)):]

    # To store the IDs of patients
    patient_ids_unhealthy_train = [element[:10] for element in unhealthy_train]
    patient_ids_unhealthy_val = [element[:10] for element in unhealthy_val]
    patient_ids_healthy_train = [element[:10] for element in healthy_train]
    patient_ids_healthy_val = [element[:10] for element in healthy_val]

    # Returns a list of records common to both train and val for healthy and unhealthy
    intersection_unhealthy = intersection(patient_ids_unhealthy_train, patient_ids_unhealthy_val)
    intersection_healthy = intersection(patient_ids_healthy_train, patient_ids_healthy_val)

    # Move half of the common unhealthy to train and other half to val
    move_to_train = intersection_unhealthy[:int(0.5*len(intersection_unhealthy))]
    move_to_val = intersection_unhealthy[int(0.5*len(intersection_unhealthy)):]
    # print("move_to_train: ", len(move_to_train))
    # print("move_to_val: ", len(move_to_val))

    for patient_id in move_to_train:
        in_val = []
        
        # find and remove all files in val common to train
        for file_ in unhealthy_val:
            if file_[:10] == patient_id:
                in_val.append(file_)
                unhealthy_val.remove(file_)
                
        # add to train
        for file_ in in_val:
            unhealthy_train.append(file_)
            
    for patient_id in move_to_val:
        in_train = []
        
        # find and remove all files in train common to val
        for file_ in unhealthy_train:
            if file_[:10] == patient_id:
                in_train.append(file_)
                unhealthy_train.remove(file_)
                
        # add to train
        for file_ in in_train:
            unhealthy_val.append(file_)

    # Similarly for healthy
    move_to_train = intersection_healthy[:int(0.5*len(intersection_healthy))]
    move_to_val = intersection_healthy[int(0.5*len(intersection_healthy)):]

    for patient_id in move_to_train:
        in_val = []
        
        # find and remove all files in val
        for file_ in healthy_val:
            if file_[:10] == patient_id:
                in_val.append(file_)
                healthy_val.remove(file_)
                
        # add to train
        for file_ in in_val:
            healthy_train.append(file_)
            

    for patient_id in move_to_val:
        in_train = []
        
        # find and remove all files in val
        for file_ in healthy_train:
            if file_[:10] == patient_id:
                in_train.append(file_)
                healthy_train.remove(file_)
                
        # add to train
        for file_ in in_train:
            healthy_val.append(file_)

    # Reading data from channels: channel_1 = v1, channel_2 = v6, for healthy patients
    data_healthy_train = []
    for file in healthy_train:
        data_v4, _ = wfdb.rdsamp(folder_name + file[:-1], channel_names=[str(channel_1)])
        # data_v5, _ = wfdb.rdsamp("ptbdb_data/" + file[:-1], channel_names=[str(channel_2)])
        data_v5 = np.arange(0)
        data = [data_v4.flatten(), data_v5.flatten()]
        data_healthy_train.append(data)
    data_healthy_val = []
    for file in healthy_val:
        data_v4, _ = wfdb.rdsamp(folder_name + file[:-1], channel_names=[str(channel_1)])
        # data_v5, _ = wfdb.rdsamp("ptbdb_data/" + file[:-1], channel_names=[str(channel_2)])
        data_v5 = np.arange(0)
        data = [data_v4.flatten(), data_v5.flatten()]
        data_healthy_val.append(data)

    # Reading data from channels: channel_1 = v1, channel_2 = v6, for unhealthy patients
    data_unhealthy_train = []
    for file in unhealthy_train:
        data_v4, _ = wfdb.rdsamp(folder_name + file[:-1], channel_names=[str(channel_1)])
        # data_v5, _ = wfdb.rdsamp("ptbdb_data/" + file[:-1], channel_names=[str(channel_2)])
        data_v5 = np.arange(0)
        data = [data_v4.flatten(), data_v5.flatten()]
        data_unhealthy_train.append(data)
    data_unhealthy_val = []
    for file in unhealthy_val:
        data_v4, _ = wfdb.rdsamp(folder_name + file[:-1], channel_names=[str(channel_1)])
        # data_v5, _ = wfdb.rdsamp("ptbdb_data/" + file[:-1], channel_names=[str(channel_2)])
        data_v5 = np.arange(0)
        data = [data_v4.flatten(), data_v5.flatten()]
        data_unhealthy_val.append(data)

    # Converting the list of channels data to array

    # Calling of function    
    data_healthy_train = toReducedArray(data_healthy_train)
    data_healthy_val = toReducedArray(data_healthy_val)
    data_unhealthy_train = toReducedArray(data_unhealthy_train)
    data_unhealthy_val = toReducedArray(data_unhealthy_val)

    print("data_healthy_train shape: ", data_healthy_train.shape)
    print("data_healthy_val shape: ", data_healthy_val.shape)
    print("data_unhealthy_train shape: ", data_unhealthy_train.shape)
    print("data_unhealthy_val shape: ", data_unhealthy_val.shape)

    return data_healthy_train, data_healthy_val, data_unhealthy_train, data_unhealthy_val

#############################################################

def get_batch(data_unhealthy_train, data_healthy_train, data_unhealthy_val,
                data_healthy_val, batch_size, split='train'):

    if split == 'train':
        # batch_size = min(batch_size, min(data_healthy_train.shape[0], data_unhealthy_train.shape[0]))
        # unhealthy_indices_np = np.random.choice(np.arange(len(data_unhealthy_train)), size=int(batch_size / 2), replace=False)
        unhealthy_indices_np = np.random.choice(np.arange(len(data_unhealthy_train)), size=int(batch_size / 2), replace=True)
        unhealthy_indices = unhealthy_indices_np.tolist()
        
        # healthy_indices_np = np.random.choice(np.arange(len(data_healthy_train)), size=int(batch_size / 2), replace=False)
        healthy_indices_np = np.random.choice(np.arange(len(data_healthy_train)), size=int(batch_size / 2), replace=True)
        healthy_indices = healthy_indices_np.tolist()
        
        unhealthy_batch = data_unhealthy_train[unhealthy_indices]
        healthy_batch = data_healthy_train[healthy_indices]
    elif split == 'val':
        # batch_size = min(batch_size, min(data_healthy_val.shape[0], data_unhealthy_val.shape[0])) 
        # unhealthy_indices_np = np.random.choice(np.arange(len(data_unhealthy_val)), size=int(batch_size / 2), replace=False)
        unhealthy_indices_np = np.random.choice(np.arange(len(data_unhealthy_val)), size=int(batch_size / 2), replace=True)
        unhealthy_indices = unhealthy_indices_np.tolist()
        
        # healthy_indices_np = np.random.choice(np.arange(len(data_healthy_val)), size=int(batch_size / 2), replace=False)
        healthy_indices_np = np.random.choice(np.arange(len(data_healthy_val)), size=int(batch_size / 2), replace=True)
        healthy_indices = healthy_indices_np.tolist()
        
        unhealthy_batch = data_unhealthy_val[unhealthy_indices]
        healthy_batch = data_healthy_val[healthy_indices]
        
    batch_x = []
    for sample in unhealthy_batch:
        
        start = random.choice(np.arange(len(sample[0]) - window_size))
        
        # normalize
        normalized_1 = minmax_scale(sample[0][start:start+window_size])
        normalized_2 = minmax_scale(sample[1][start:start+window_size])
        normalized = np.array((normalized_1, normalized_2))
        
        batch_x.append(normalized)
        
    for sample in healthy_batch:
        
        start = random.choice(np.arange(len(sample[0]) - window_size))
        
        # normalize
        normalized_1 = minmax_scale(sample[0][start:start+window_size])
        normalized_2 = minmax_scale(sample[1][start:start+window_size])
        normalized = np.array((normalized_1, normalized_2))
        
        batch_x.append(normalized)
    
    batch_y = [0.1 for _ in range(int(batch_size / 2))]
    for _ in range(int(batch_size / 2)):
        batch_y.append(0.9)
        
    indices = np.arange(len(batch_y))
    np.random.shuffle(indices)
    
    batch_x = np.array(batch_x)
    batch_y = np.array(batch_y)
    
    batch_x = batch_x[indices]
    batch_y = batch_y[indices]
    
    batch_x = np.reshape(batch_x, (-1, 2, window_size))
    batch_x = torch.from_numpy(batch_x)
    # batch_x = batch_x.float().cuda()
    batch_x = batch_x.float()
    batch_x = batch_x.float()
    
    batch_y = np.reshape(batch_y, (-1, 1))
    batch_y = torch.from_numpy(batch_y)
    # batch_y = batch_y.float().cuda()
    batch_y = batch_y.float()
    batch_y = batch_y.float()
    
    return batch_x, batch_y 

#############################################################

def validate_local_model(model, data_unhealthy_val, data_healthy_val, batch_size = 10):
    # Validating the model
    data_unhealthy_train = np.zeros((1, 2, 1))
    data_healthy_train = np.zeros((1, 2, 1))

    iterations = 100
    overall_correct = 0

    with torch.inference_mode():        # To disable gradients and forward mode
        for _ in range(iterations):
            batch_x, batch_y = get_batch(data_unhealthy_train, data_healthy_train, 
                                            data_unhealthy_val, data_healthy_val, batch_size, 
                                                split='val')
            y_pred = model(batch_x)
            
            no_correct = 0
            acc = 0
            i = 0
            for num in y_pred:
                if int(torch.round(num)) == int(torch.round(batch_y[i])):
                    acc += 10
                    no_correct += 1
                i += 1
            
            overall_correct += no_correct
        
    total = batch_size * iterations
    accuracy = overall_correct / total
    print(f"Accuracy after local training: {accuracy:.2%}")

    return(accuracy)

#############################################################

# Function to train the local model
def train_local_model(model, data_unhealthy_train, data_healthy_train, 
                        data_unhealthy_val, data_healthy_val, server_control, lr=1.0e-4, epochs=5, 
                            batch_size = 10):
    writer = SummaryWriter('/Users/Ryan/Desktop/Final Year Project/federated_learning/runs/runs_cli' 
                            + str(client_no) + '_' + str(channel_1) + '_it' + str(fed_iteration_no))

    model_before_train = copy.deepcopy(model)

    # Training the model
    criterion = torch.nn.BCELoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-4)
    optimizer = ScaffoldOptimizer(model.parameters(), lr=model.lr, weight_decay=1e-4)
    lr_step = StepLR(optimizer, step_size=10, gamma=0.1)

    for epoch in range(epochs):
        batch_x, batch_y = get_batch(data_unhealthy_train, data_healthy_train, 
                                        data_unhealthy_val, data_healthy_val, batch_size, 
                                            split='train')
        
        # Forward pass
        y_pred = model(batch_x)
        loss = criterion(y_pred, batch_y)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step(server_control, model.control)

        print(f"Iteration {epoch} complete!")

        if (epoch % 100) == 0 and epoch != 0:
            lr_step.step()
            accuracy = validate_local_model(model, data_unhealthy_val, data_healthy_val, batch_size)
            writer.add_scalar('Accuracy/val', (accuracy * 100), epoch)

    # update c
    # c+ <- ci - c + 1/(steps * lr) * (x-yi)
    # save ann
    temp = {}
    for k, v in model.named_parameters():
        temp[k] = v.data.clone()

    for k, v in model_before_train.named_parameters():
        local_steps = model.E * batch_size      # = epochs * batch_size
        model.control[k] = model.control[k] - server_control[k] + (v.data - temp[k]) / (local_steps * model.lr)
        model.delta_y[k] = temp[k] - v.data
        model.delta_control[k] = model.control[k] - model_before_train.control[k]

    return model

# Function to run the client
def run_client():
    try:
        # Initialize client socketx
    #    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    #    client_socket.connect(('localhost', 8080))

        K, C, B, r = 10, 0.5, 50, 10
        input_dim = 32
        lr = 0.08

        epochs = E = 30000          # E = 30000

        # Initialize model
        model = ConvNetQuake(B, epochs, lr, name = f'client_{client_no}')
        #model = nn.DataParallel(model, device_ids=[0])

        # Getting global model parameters from locally stored file
        if fed_iteration_no != 1:
            model_dict = torch.load('server_model_parameters.pth')
            model.load_state_dict(model_dict)
            print("Loaded global model parameters from server_model_parameters.pth")

            # Loading client control related variables
            with open(f'cli{client_no}_control.pkl', 'rb') as fc:
                model.control = pickle.load(fc)
            with open(f'cli{client_no}_delta_control.pkl', 'rb') as fdc:
                model.delta_control = pickle.load(fdc)
            with open(f'cli{client_no}_delta_y.pkl', 'rb') as fdy:
                model.delta_y = pickle.load(fdy)
            print("Loaded control, delta_control and delta_y.")

            # Loading server control variables
            with open(f'server_control.pkl', 'rb') as fsc:
                server_control = pickle.load(fsc)
            print("Loaded server_control.")
        else:
            server_control = {}
            for k, v in model.named_parameters():
                server_control[k] = torch.zeros_like(v.data)

        print("\nLoading data for training.")
        data_healthy_train, data_healthy_val, data_unhealthy_train, data_unhealthy_val = load_data()
    
        # Train local model
        print("\nTraining the local model.")
        model = train_local_model(model, data_unhealthy_train, data_healthy_train, 
                                    data_unhealthy_val, data_healthy_val, server_control, lr=model.lr, epochs=30001, 
                                        batch_size = 10)

        # Prepare data to be sent to server
    #    data_to_send = {
    #                    'conv1.weight': model.conv1.weight.detach().numpy(),
    #                    'conv1.bias': model.conv1.bias.detach().numpy(),
    #                    'conv2.weight': model.conv2.weight.detach().numpy(),
    #                    'conv2.bias': model.conv2.bias.detach().numpy(),
    #                    'conv3.weight': model.conv3.weight.detach().numpy(),
    #                    'conv3.bias': model.conv3.bias.detach().numpy(),
    #                    'conv4.weight': model.conv4.weight.detach().numpy(),
    #                    'conv4.bias': model.conv4.bias.detach().numpy(),
    #                    'conv5.weight': model.conv5.weight.detach().numpy(),
    #                    'conv5.bias': model.conv5.bias.detach().numpy(),
    #                    'conv6.weight': model.conv6.weight.detach().numpy(),
    #                    'conv6.bias': model.conv6.bias.detach().numpy(),
    #                    'conv7.weight': model.conv7.weight.detach().numpy(),
    #                    'conv7.bias': model.conv7.bias.detach().numpy(),
    #                    'conv8.weight': model.conv8.weight.detach().numpy(),
    #                    'conv8.bias': model.conv8.bias.detach().numpy()
    #                    }

        # Send data to server
    #    client_socket.send(pickle.dumps(data_to_send))
    #    print("/nSent model parameters to the server.")

    #    client_socket.close()
        
        # Storing the weights locally in a file
        torch.save(model.state_dict(), f'cli{client_no}_model_parameters.pth')
        print(f"Saved local model parameters in cli{client_no}_model_parameters.pth")

        # Storing control related variables
        with open(f'cli{client_no}_control.pkl', 'wb') as fc:
            pickle.dump(model.control, fc)
        with open(f'cli{client_no}_delta_control.pkl', 'wb') as fdc:
            pickle.dump(model.delta_control, fdc)
        with open(f'cli{client_no}_delta_y.pkl', 'wb') as fdy:
            pickle.dump(model.delta_y, fdy)
        print(f'Storing control, delta_control and delta_y in cli{client_no}_control.pkl, cli{client_no}_delta_control.pkl and cli{client_no}_delta_y.pkl respectively')

    except ConnectionRefusedError:
        print("Connection refused. Server might be offline.")
    except Exception as e:
        print("An error occurred:", e)

if __name__ == "__main__":
    run_client()
