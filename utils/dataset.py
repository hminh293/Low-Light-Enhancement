import numpy as np
import cv2
import PIL.Image as Image
import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torchvision.datasets as datasets
import torchvision.utils as utils
import torchvision.transforms.functional as TF
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader
import random

def load_image(file_path): # folder
    folder_list = os.listdir(file_path)
    folder_data = {}
    for folder_name in folder_list:
        folder_data[folder_name] = {}
        image_list = os.listdir(os.path.join(file_path, folder_name))
        folder_dir = os.path.join(file_path, folder_name)
        for image_name in image_list:
            image_path = os.path.join(folder_dir, image_name)
            image = Image.open(image_path)
            image = image.convert('RGB')
            image = image.resize((256, 256))
            image = np.array(image)
            image = image.astype(np.float32)
            image = image / 255.0
            image = image.transpose(2, 0, 1)
            folder_data[folder_name][image_name] = image
    return folder_data

def create_low_image(folder_data, k = [0.3, 0.7] ,alpha = [1.5,2.5]): #folder_data is a dictionary of images folder (train/test/val)
    low_image_folder = {}
    for folder_name, image_folder in folder_data.items():
        low_image_folder[folder_name] = {}
        for name, image in image_folder.items():
            k_random = np.random.uniform(k[0], k[1])
            alpha_random = np.random.uniform(alpha[0], alpha[1])
            low_image = k_random * (image ** alpha_random)
            low_image_folder[folder_name][name] = low_image
    return low_image_folder

def combine_image(high_image_data, low_image_data): #high_image_data is a dictionary of images_folder (train/test/val), low_image_data is a dictionary of low images_folder (train/test/val)
    combined_image_folder = {}
    for folder_name, image_folder in high_image_data.items():
        combined_image_folder[folder_name] = {}
        for name, image in image_folder.items():
            combined_image_folder[folder_name][name] = [image, low_image_data[folder_name][name]]
    return combined_image_folder

def data_create(folder_path):
    print("Loading data...")
    folder_data = load_data(folder_path)
    print("Creating low image...")
    low_image_data = create_low_image(folder_data)
    print("Combining image...")
    combined_image_data = combine_image(folder_data, low_image_data)
    print("Data created successfully")
    return combined_image_data


class LowLightDataset(Dataset):
    def __init__(self, root_dir, mode = 'train') -> None:
        super().__init__()
        self.root_dir = root_dir
        self.mode = mode
        self.images_path = []
        search_path = os.path.join(root_dir, mode, '*.jpg')
        self.images_path = glob.glob(search_path)

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, index):
        light_image_path = self.images_path[index]
        
        light_image = Image.open(light_image_path)
        light_image = light_image.convert('RGB')
        light_image = light_image.resize((256, 256))
        light_image = np.array(light_image)
        light_image = light_image.astype(np.float32)
        light_image = light_image / 255.0
        light_image = light_image.transpose(2, 0, 1)
        
        k = random.uniform(0.3, 0.7)
        alpha = random.uniform(1.5, 2.5)
        low_image = k * np.power(light_image, alpha)
        
        return torch.from_numpy(low_image), torch.from_numpy(light_image)
    
def get_loader(root_dir, mode = 'train', batch_size = 16, num_workers = 4):
    dataset = LowLightDataset(root_dir, mode)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

if __name__ == "__main__":
    root_dir = './dataset'
    if not os.path.exists(root_dir):
        print("Dataset not found")
        exit()
    mode = 'train'
    batch_size = 16
    num_workers = 4
    loader = get_loader(root_dir, mode, batch_size, num_workers)
    light_image, low_image = next(iter(loader))
    print(light_image.shape)
    print(low_image.shape)
        