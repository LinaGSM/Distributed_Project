import torchvision
import torch
import wget
import zipfile
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torchvision import transforms
from utils import experiment_not_implemented_message
from os.path import isdir


data_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,],
                             std=[0.229,])
    ])


class TitanicData(torch.utils.data.Dataset):
    def __init__(self, nom_fichier, is_training=True):
        data = pd.read_csv(nom_fichier)
        data.Sex = data.Sex.astype('category').cat.codes.astype("int8")
        data.Embarked = data.Embarked.astype('category').cat.codes.astype("int8")
        data.describe()
        data.Age.fillna(int(data.Age.mean()), inplace=True)
        data.Embarked.fillna(int(data.Embarked.mean()), inplace=True)
        df_train = data.drop(['Name', 'Ticket', 'Cabin', 'PassengerId'], axis=1)
        df_train.Age = MinMaxScaler().fit_transform(np.array(df_train.Age).reshape(-1, 1))
        df_train.Fare = MinMaxScaler().fit_transform(np.array(df_train.Fare).reshape(-1,1))
        numpy_y = np.array(df_train["Survived"])
        numpy_x = np.array(df_train.drop("Survived", axis=1))
        x_train, x_test, y_train, y_test = train_test_split(numpy_x, numpy_y,
                                                            test_size=0.2, random_state=1)
        if is_training:
            self.X = torch.tensor(x_train, dtype=torch.float32)
            self.label = torch.unsqueeze(torch.tensor(y_train, dtype=torch.float32), dim=-1)
        else:
            self.X = torch.tensor(x_test, dtype=torch.float32)
            self.label = torch.unsqueeze(torch.tensor(y_test, dtype=torch.float32), dim=-1)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, indice):
        return self.X[indice], self.label[indice]


def get_loader(experiment_name, batch_size=1, is_trainning = True):
    """
    constructs data loader for an experiment

    Parameters
    ----------
    experiment_name: str
        name of the experiment to be used;
        possible are {"mnist"}

    batch_size: int
        the size of the batch

    is_trainning: bool
        the use of training dataset

    Returns
    -------
        DataLoader

    """
    if experiment_name == "faces":
        if not isdir("data/faces/training"):
            url = "http://www-sop.inria.fr/members/Chuan.Xu/faces.zip"
            wget.download(url)
            with zipfile.ZipFile("faces.zip", "r") as zip_ref:
                zip_ref.extractall("./data")
        if is_trainning:
            pathname = "data/faces/training"
        else:
            pathname = "data/faces/testing"
        dataset = torchvision.datasets.ImageFolder(pathname, transform=data_transform)

    elif experiment_name == "fash_mnist":
        if is_trainning:
            dataset = torchvision.datasets.FashionMNIST('data/', train=True, transform=data_transform, download=True)
        else:
            dataset = torchvision.datasets.FashionMNIST('data/', train=False, transform=data_transform, download=True)

    #elif experiment_name == "titanic":
    #    dataset = TitanicData("data/titanic/train.csv", is_training=is_trainning)
    else:
        raise NotImplementedError(
            experiment_not_implemented_message(experiment_name=experiment_name)
        )

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    return data_loader