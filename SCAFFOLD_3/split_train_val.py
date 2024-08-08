import numpy as np

seed_num = 30
run_num = 1
folder_names = ['ptbdb_data_cli1/', 'ptbdb_data_cli2/']
train_val_split_percent = 0.8

def run_program():
    np.random.seed(int(seed_num))
    files_unhealthy_train, files_unhealthy_val, files_healthy_train, files_healthy_val = [], [], [], []

    for name in folder_names:
        with open(name + 'RECORDS') as fp:  
            lines = fp.readlines()
        
        files_unhealthy, files_healthy = [], []

        # To find out files with healthy and unhealthy patients(suffering from myocardial infarction)
        for file in lines:
            file_path = name + file[:-1] + ".hea"
            
            # read header to determine class
            if 'Myocardial infarction' in open(file_path).read():
                files_unhealthy.append(file)
                
            if 'Healthy control' in open(file_path).read():
                files_healthy.append(file)

        files_unhealthy_train.extend(files_unhealthy[:int(train_val_split_percent*len(files_unhealthy))])
        files_unhealthy_val.extend(files_unhealthy[int(train_val_split_percent*len(files_unhealthy)):])
        files_healthy_train.extend(files_healthy[:int(train_val_split_percent*len(files_healthy))])    
        files_healthy_val.extend(files_healthy[int(train_val_split_percent*len(files_healthy)):])

    # To store the names of files that are healthy_train, healthy_val
    # unhealthy_train and unhealthy_val in their respective files.
    file_names = ["unhealthy_train", "unhealthy_val", "healthy_train", "healthy_val"]
    list_names = [files_unhealthy_train, files_unhealthy_val, files_healthy_train, files_healthy_val]
    for i in range(4):
        with open(file_names[i] + '.txt', 'w') as fp:
            for record in list_names[i]:
                fp.write(record)


if __name__ == "__main__":
    run_program()