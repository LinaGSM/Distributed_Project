import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import time
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision
from torch.utils.data import DataLoader

from torch.nn.parallel import DistributedDataParallel as DDP

import csv

def train_model(model, data_loader, optimizer, loss_function, batch_size, num_epochs=5):
    with open('training_time.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        
        # Ajouter les en-têtes si le fichier est vide
        if file.tell() == 0:
            writer.writerow(['Loading Time', 'Computation + Communication Time', 'Real Time', 'Batch Size'])
        
        start = time.time()  # Temps de début total
        
        # Temps de chargement des données
        start_loading_time = time.time()
        train_images, train_labels = next(iter(dataloader))
        loading_time = time.time() - start_loading_time
        print(f'Loading time: {loading_time} seconds')
        
        # Temps de calcul et de communication
        start_comp_comm_time = time.time()
        optimizer.zero_grad()
        outputs = ddp_model(train_images)
        loss_fn(outputs, train_labels).backward()
        computation_communication_time = time.time() - start_comp_comm_time
        
        real_time = time.time() - start  # Temps total réel
        
        # Écriture des données dans le fichier CSV
        writer.writerow([loading_time, computation_communication_time, real_time, batch_size])
        print(f"Epoch {epoch}/{num_epochs} - Loading: {loading_time:.2f}s, Computation+Comm: {computation_communication_time:.2f}s, Total: {real_time:.2f}s")

# Exemple d'appel de la fonction avec un modèle, un data_loader et un optimizer

    # Assurez-vous de définir `model`, `data_loader`, `optimizer`, `loss_function`, et `batch_size` avant d'appeler `train_model`


def run(rank, size):
    # Define transformations
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = torchvision.datasets.Imagenette('/data/neo/user/chxu/', transform=transform_train)
    dataset_size = len(dataset)
    localdataset_size = dataset_size//size
    local_dataset = torch.utils.data.Subset(dataset, range(rank*localdataset_size, (rank+1)*localdataset_size))
    sample_size = 32//size
    #batch_size=sample_size
    dataloader = DataLoader(local_dataset, batch_size=sample_size, shuffle=True)
    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, len(dataset.classes))
    ddp_model = DDP(model)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)


    with open('training_time.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        
        # Ajouter les en-têtes si le fichier est vide
        if file.tell() == 0:
            writer.writerow(['Loading Time', 'Computation + Communication Time', 'Real Time', 'Batch Size'])
        
        start = time.time()  # Temps de début total
        
        # Temps de chargement des données
        start_loading_time = time.time()
        train_images, train_labels = next(iter(dataloader))
        loading_time = time.time() - start_loading_time
        print(f'Loading time: {loading_time} seconds')
        
        # Temps de calcul et de communication
        start_comp_comm_time = time.time()
        optimizer.zero_grad()
        outputs = ddp_model(train_images)
        loss_fn(outputs, train_labels).backward()
        computation_communication_time = time.time() - start_comp_comm_time
        
        real_time = time.time() - start  # Temps total réel
        
        optimizer.step()
        dist.destroy_process_group()
        print(f"Finished running basic DDP example on rank {rank}.")
        print(f"Total time: {real_time} seconds")

        # Écriture des données dans le fichier CSV
        writer.writerow([loading_time, computation_communication_time, real_time])
        print(f" Loading: {loading_time}s, Computation+Comm: {computation_communication_time}s, Total: {real_time}s")

'''
    print(f"Start running basic DDP example on rank {rank} with model Resnet18.")
    st = time.time()
    train_images, train_labels = next(iter(dataloader))
    et_read = time.time()
    print(f'Loading time: {et_read-st} seconds')
    optimizer.zero_grad()
    outputs = ddp_model(train_images)
    loss_fn(outputs, train_labels).backward()
    et = time.time()
    print(f'Computing + Communication time: {et-et_read} seconds')
    optimizer.step()
    dist.destroy_process_group()
    print(f"Finished running basic DDP example on rank {rank}.")
    print(f"Total time: {et-st} seconds")
'''

if __name__ == "__main__":
    dist.init_process_group("gloo", init_method="env://")
    size = dist.get_world_size()
    rank = dist.get_rank()
    run(rank, size)