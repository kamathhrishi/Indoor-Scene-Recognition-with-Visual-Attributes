import scipy.io as sio
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from torch.utils.data.sampler import SubsetRandomSampler
import torch
import numpy as np
from sklearn.model_selection import train_test_split


def shuffle_loader(data, shuffle_dataset=True, random_seed=42, drop_last=False):

    dataset_size = len(data)
    indices = list(range(dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    # Creating PT data samplers and loaders:
    sampler = SubsetRandomSampler(indices)

    loader = torch.utils.data.DataLoader(
        data, batch_size=16, sampler=sampler, drop_last=drop_last
    )

    return loader


class Dataset(Dataset):
    def __init__(self, data, labels, transform):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):

        return len(self.data)

    def __getitem__(self, idx):

        return self.transform(self.data[idx]), self.labels[idx]


def load_dataset():

    mat_contents = sio.loadmat("SUNAttributeDB/attributeLabels_continuous.mat")
    attr = sio.loadmat("SUNAttributeDB/attributeLabels_continuous.mat")
    images = sio.loadmat("SUNAttributeDB/images.mat")

    images_attr = {}

    for i in range(0, len(images["images"])):

        images_attr[images["images"][i][0][0]] = attr["labels_cv"][i]

    dataset_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    IMG_SIZE = 150

    images = []
    attr = []

    for i in images_attr.keys():

        images.append(
            np.asarray(
                Image.open("Images" + "/" + i)
                .convert("RGB")
                .resize((IMG_SIZE, IMG_SIZE))
            )
        )
        attr.append((images_attr[i] > 0).astype(int))

    X_train, X_test, y_train, y_test = train_test_split(
        images, attr, test_size=0.33, random_state=42
    )

    train_dataset = Dataset(X_train, y_train, dataset_transform)
    train_dataloader = shuffle_loader(train_dataset)

    test_dataset = Dataset(X_test, y_test, dataset_transform)
    test_dataloader = shuffle_loader(test_dataset)

    return train_dataloader, test_dataloader
