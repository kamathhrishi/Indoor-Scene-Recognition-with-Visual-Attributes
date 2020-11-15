# Import required libraries
import os
import numpy as np
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from arguments import Arguments

args = Arguments()


def segtext(type: str):
    # Get names of images present in training/test set from text file

    file = open(type + ".txt")
    Tags = {}

    for line in file.readlines():
        file_info = line.split("/")
        Tags[file_info[1].rstrip()] = file_info[0]

    return Tags


def shuffle_loader(data, shuffle_dataset=True, random_seed=42):

    dataset_size = len(data)
    indices = list(range(dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    # Creating PT data samplers and loaders:
    sampler = SubsetRandomSampler(indices)

    loader = torch.utils.data.DataLoader(
        data, batch_size=args.batch_size, sampler=sampler
    )

    return loader


class SceneDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, scenes, labels, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.transform = transform
        self.scenes = scenes
        self.labels = labels

    def __len__(self):

        return len(self.scenes)

    def __getitem__(self, idx):

        # print(self.scenes[idx].shape)
        return self.transform(self.scenes[idx]), self.labels[idx]


def data_transform(img, IMG_SIZE):
    # return np.transpose(np.asarray(img.resize((IMG_SIZE,IMG_SIZE))),(2,1,0))
    return img.resize((IMG_SIZE, IMG_SIZE))


def load_image(path, IMG_SIZE):

    im = Image.open(path)
    im = np.asarray(im.resize((IMG_SIZE, IMG_SIZE)))

    temp_img = ""
    temp_label = ""

    if len(im.shape) == 3 and im.shape[2] == 3:

        # im=np.transpose(im,(2,1,0))
        temp_img = im
        temp_label = im

    return temp_img, temp_label


def load_data(data_type: str, IMG_SIZE, torch_transform):

    Images = []
    Labels = []

    Label_to_ix = {}
    index = 0

    for i in os.listdir(data_type):

        if i != ".DS_Store":

            if i not in Label_to_ix:

                Label_to_ix[i] = index
                index += 1

            for j in os.listdir(data_type + "/" + i):

                if j != ".DS_Store":

                    image, label = load_image(data_type + "/" + i + "/" + j, IMG_SIZE)

                    if type(image) != str and type(label) != str:

                        img = Image.open(data_type + "/" + i + "/" + j)
                        # cropped_imgs=crop_img(img,IMG_SIZE)

                        img_set = [img]

                        for u in img_set:

                            Images.append(data_transform(u, IMG_SIZE))
                            Labels.append(Label_to_ix[i])

                        # print(len(Images))
    dataset = SceneDataset(Images, Labels, transform=torch_transform)
    loader = shuffle_loader(dataset)

    return loader


def load_dataset():

    traindata_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    testdata_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_loader = load_data("train", 224, traindata_transform)
    test_loader = load_data("test", 224, testdata_transform)

    return train_loader, test_loader
