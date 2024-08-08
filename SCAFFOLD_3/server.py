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
import torch.nn as nn
import socket
import pickle

channel_1 = 'v6'
channel_2 = 'vz'
seed_num = 30
run_num = 1
folder_names = ['ptbdb_data_cli1/', 'ptbdb_data_cli2/']
health_train_val_files = ["unhealthy_train", "unhealthy_val", "healthy_train", "healthy_val"]
no_of_clients = 2
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

# Returns a list of records common to both lst1 and lst2
def intersection(lst1, lst2): 
    return list(set(lst1) & set(lst2))

# Fix problems with array shape
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
            
    #print("Reduced data array:-\n", data_reduced)
    return(data_reduced)

def get_batch(data_unhealthy_train, data_healthy_train, data_unhealthy_val,
                data_healthy_val, batch_size, split='train'):
    if split == 'train':
        unhealthy_indices_np = np.random.choice(np.arange(len(data_unhealthy_train)), size=int(batch_size / 2), replace=False)
        unhealthy_indices = unhealthy_indices_np.tolist()
        
        healthy_indices_np = np.random.choice(np.arange(len(data_healthy_train)), size=int(batch_size / 2), replace=False)
        healthy_indices = healthy_indices_np.tolist()
        
        unhealthy_batch = data_unhealthy_train[unhealthy_indices]
        healthy_batch = data_healthy_train[healthy_indices]
    elif split == 'val': 
        unhealthy_indices_np = np.random.choice(np.arange(len(data_unhealthy_val)), size=int(batch_size / 2), replace=False)
        unhealthy_indices = unhealthy_indices_np.tolist()
        
        healthy_indices_np = np.random.choice(np.arange(len(data_healthy_val)), size=int(batch_size / 2), replace=False)
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

def load_data():
# ---------------------------------------------------------------------- #   
    # Previous way for of loading data by simply reading the records and splitting

    # files_unhealthy, files_healthy = [], []

    # for name in folder_names:
    #     with open(name + 'RECORDS') as fp:  
    #         lines = fp.readlines()
        
        # To find out files with healthy and unhealthy patients(suffering from myocardial infarction)
    #    for file in lines:
    #        file_path = name + file[:-1] + ".hea"
            
            # read header to determine class
    #        if 'Myocardial infarction' in open(file_path).read():
    #            files_unhealthy.append(file)
                
    #        if 'Healthy control' in open(file_path).read():
    #            files_healthy.append(file)

    # shuffle data (cross-validation)
    # np.random.seed(int(seed_num))
    # np.random.shuffle(files_unhealthy)
    # np.random.shuffle(files_healthy)

    # healthy_train = files_healthy[:int(0.8*len(files_healthy))]
    # healthy_val = files_healthy[int(0.8*len(files_healthy)):]
    # unhealthy_train = files_unhealthy[:int(0.8*len(files_unhealthy))]
    # unhealthy_val = files_unhealthy[int(0.8*len(files_unhealthy)):]
# ---------------------------------------------------------------------- #
    i = 0
    unhealthy_train, unhealthy_val, healthy_train, healthy_val = [], [], [], []

    for file in health_train_val_files:
        with open(file + '.txt') as fp:  
            lines = fp.readlines()
        
        if i == 0:
            unhealthy_train.extend(lines)
        elif i == 1:
            unhealthy_val.extend(lines)
        elif i == 2:
            healthy_train.extend(lines)
        elif i == 3:
            healthy_val.extend(lines)
        
        i += 1

    # To store the IDs of patients
    patient_ids_unhealthy_train = [element[:10] for element in unhealthy_train]
    patient_ids_unhealthy_val = [element[:10] for element in unhealthy_val]
    patient_ids_healthy_train = [element[:10] for element in healthy_train]
    patient_ids_healthy_val = [element[:10] for element in healthy_val]
    #  print('Unhealthy train patients: ', len(patient_ids_unhealthy_train))
    #  print('Unhealthy val patients: ', len(patient_ids_unhealthy_val))
    #  print('Healthy train patients: ', len(patient_ids_healthy_train))
    #  print('Healthy val patients: ', len(patient_ids_healthy_val))

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
    data_healthy_val = []
    i = 0
    j = 0
    for name in folder_names:
        # for file in healthy_train:
        while(i < len(healthy_train)):
            file_name = name + healthy_train[i][:-1]
            # print("file_name = ", file_name)
            if os.path.exists(file_name + '.dat'):
                # print("file exists")
                data_v4, _ = wfdb.rdsamp(file_name, channel_names=[str(channel_1)])
                # data_v5, _ = wfdb.rdsamp("ptbdb_data/" + file[:-1], channel_names=[str(channel_2)])
                data_v5 = np.arange(0)
                data = [data_v4.flatten(), data_v5.flatten()]
                data_healthy_train.append(data)
            else:
                break
            i += 1
        
        while(j < len(healthy_val)):
        # for file in healthy_val:
            file_name = name + healthy_val[j][:-1]
            if os.path.exists(file_name + '.dat'):
                data_v4, _ = wfdb.rdsamp(file_name, channel_names=[str(channel_1)])
                # data_v5, _ = wfdb.rdsamp("ptbdb_data/" + file[:-1], channel_names=[str(channel_2)])
                data_v5 = np.arange(0)
                data = [data_v4.flatten(), data_v5.flatten()]
                data_healthy_val.append(data)
            else:
                break
            j += 1

    # Reading data from channels: channel_1 = v1, channel_2 = v6, for unhealthy patients
    data_unhealthy_train = []
    data_unhealthy_val = []
    i = 0
    j = 0
    for name in folder_names:
    # for file in unhealthy_train:
        while(i < len(unhealthy_train)):
            file_name = name + unhealthy_train[i][:-1]
            if os.path.exists(file_name + '.dat'):
                data_v4, _ = wfdb.rdsamp(file_name, channel_names=[str(channel_1)])
                # data_v5, _ = wfdb.rdsamp("ptbdb_data/" + file[:-1], channel_names=[str(channel_2)])
                data_v5 = np.arange(0)
                data = [data_v4.flatten(), data_v5.flatten()]
                data_unhealthy_train.append(data)
            else:
                break
            i += 1
    # for file in unhealthy_val:
        while(j < len(unhealthy_val)):
            file_name = name + unhealthy_val[j][:-1]
            if os.path.exists(file_name + '.dat'):
                data_v4, _ = wfdb.rdsamp(file_name, channel_names=[str(channel_1)])
                # data_v5, _ = wfdb.rdsamp("ptbdb_data/" + file[:-1], channel_names=[str(channel_2)])
                data_v5 = np.arange(0)
                data = [data_v4.flatten(), data_v5.flatten()]
                data_unhealthy_val.append(data)
            else:
                break
            j += 1

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


# Federated learning training function
def validate_global_model(model, data_unhealthy_train, data_healthy_train, 
                            data_unhealthy_val, data_healthy_val, epochs = 5, batch_size = 10):
    overall_count = 0
    
    with torch.inference_mode():
        for _ in range(epochs):
            batch_x, batch_y = get_batch(data_unhealthy_train, data_healthy_train, 
                                            data_unhealthy_val, data_healthy_val, batch_size, 
                                                split='val')

            # Finding predictions
            y_pred = model(batch_x)

            count = 0
            acc = 0
            for num in y_pred:
                if int(torch.round(num)) == int(torch.round(batch_y[count])):
                    acc += 10
                    count += 1

            overall_count += count

    # Calculate accuracy
    total = batch_size * epochs
    accuracy = overall_count / total

    print(f"Accuracy of global model: {accuracy:.2%}")


#def server_aggregate(global_model, client_models):
#    global_dict = global_model.state_dict()
#    for k in global_dict.keys():
#        global_dict[k] = torch.stack([client_models[i].state_dict()[k].float() for i in range(len(client_models))], 0).mean(0)
#    global_model.load_state_dict(global_dict)
#    for model in client_models:
#        model.load_state_dict(global_model.state_dict())
    
def server_aggregate(global_model, client_models, no_sample_clients, no_total_clients):
    x = {}
    c = {}

    for k, v in client_models[0].named_parameters():
        x[k] = torch.zeros_like(v.data)
        c[k] = torch.zeros_like(v.data)

    for j in range(no_of_clients):
        for k, v in client_models[j].named_parameters():
            # x[k] += (client_models[j].delta_y[k] / no_sample_clients)         # averaging
            x[k] += (v.data / no_sample_clients)                              # Average of weights for all clients
            c[k] += (client_models[j].delta_control[k] / no_sample_clients)   # averaging

    for k, v in global_model.named_parameters():
        # v.data += x[k].data  # lr=1
        v.data = x[k].data                                                    # Storing averaged client weights in the global model 
        global_model.control[k].data += c[k].data * (no_sample_clients / no_total_clients)

    return(global_model)


def run_server():
    # Initialize model
    K, C, B, r = 10, 0.5, 50, 10
    input_dim = 32
    lr = 0.08
    epochs = E = 30000          # E = 30000

    global_model = ConvNetQuake(B, epochs, lr, name = 'server')

    while True:
        # Loading server_control if not 1st iteration
        if fed_iteration_no != 1:
            with open(f'server_control.pkl', 'rb') as fsc:
                global_model.control = pickle.load(fsc)
            print("Loaded server_control.")

        print("Loading data for validation.")
        data_healthy_train, data_healthy_val, data_unhealthy_train, data_unhealthy_val = load_data()

        # Load updated model parameters
        client_models = []
        no_of_clients = 2

        # Initialize server socket
        # server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # server_socket.bind(('localhost', 8080))
        # server_socket.listen(5)

        print("\nServer is listening...")

        for i in range(no_of_clients):
        #for _ in range (no_of_clients):
            # Accept client connection
            # print("Waiting for connection.")
            # client_socket, client_address = server_socket.accept()
            # print(f"Connection from {client_address} has been established!")

            # Receive data (model parameters) from client
            # data = []

            # while True:
            #     packet = client_socket.recv(4096)
            #     if not packet: break
            #     data.append(packet)

            # data = pickle.loads(b"".join(data))
                
            # model = ConvNetQuake()
            # model.load_state_dict(global_model.state_dict())
            # model.conv1.weight.data = torch.tensor(data['conv1.weight'])
            # model.conv1.bias.data = torch.tensor(data['conv1.bias'])
            # model.conv2.weight.data = torch.tensor(data['conv2.weight'])
            # model.conv2.bias.data = torch.tensor(data['conv2.bias'])
            # model.conv3.weight.data = torch.tensor(data['conv3.weight'])
            # model.conv3.bias.data = torch.tensor(data['conv3.bias'])
            # model.conv4.weight.data = torch.tensor(data['conv4.weight'])
            # model.conv4.bias.data = torch.tensor(data['conv4.bias'])
            # model.conv5.weight.data = torch.tensor(data['conv5.weight'])
            # model.conv5.bias.data = torch.tensor(data['conv5.bias'])
            # model.conv6.weight.data = torch.tensor(data['conv6.weight'])
            # model.conv6.bias.data = torch.tensor(data['conv6.bias'])
            # model.conv7.weight.data = torch.tensor(data['conv7.weight'])
            # model.conv7.bias.data = torch.tensor(data['conv7.bias'])
            # model.conv8.weight.data = torch.tensor(data['conv8.weight'])
            # model.conv8.bias.data = torch.tensor(data['conv8.bias'])

            # Loading client model parameters stored in files
            model = ConvNetQuake(B, epochs, lr, name = f'client_{i + 1}')
            model_dict = torch.load('cli' + str(i + 1) + '_model_parameters.pth')
            model.load_state_dict(model_dict)
            print("Loaded client ", i, " model parameters.")

            # Loading client control variables
            with open(f'cli{i + 1}_control.pkl', 'rb') as fc:
                model.control = pickle.load(fc)
            with open(f'cli{i + 1}_delta_control.pkl', 'rb') as fdc:
                model.delta_control = pickle.load(fdc)
            with open(f'cli{i + 1}_delta_y.pkl', 'rb') as fdy:
                model.delta_y = pickle.load(fdy)
            print("Loaded control, delta_control and delta_y.")

            client_models.append(model)           
            # client_socket.close()

        global_model = server_aggregate(global_model, client_models, no_of_clients, no_of_clients)
        print("\nGlobal Model Updatad!")
        
        # Printing accuracy of global model
        no_of_tests = 5
        for i in range(no_of_tests):
            print(f"Test {i}:")
            validate_global_model(global_model, data_unhealthy_train, data_healthy_train, 
                                data_unhealthy_val, data_healthy_val, epochs = 100, batch_size = 10)

        # server_socket.close()

        # Saving global model parameters locally in a file
        torch.save(global_model.state_dict(), 'server_model_parameters.pth')
        print("\nSaved global model parameters in server_model_parameters.pth")

        # Saving global model control variables in 'server_control.pkl'
        with open(f'server_control.pkl', 'wb') as fsc:
            pickle.dump(global_model.control, fsc)
        print("Saved global model control vairiables in server_control.pkl")
        break
if __name__ == "__main__":
    run_server()