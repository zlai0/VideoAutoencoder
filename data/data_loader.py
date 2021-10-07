import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import cv2
import yaml

with open('configs/datasets.yaml') as f:
    config_data = f.read()
    config = yaml.safe_load(config_data)

def image_loader(path, input_size):
    image = cv2.imread(path)
    input_h, input_w = input_size, input_size
    image = cv2.resize(image, (input_w, input_h))
    return image

def rgb_preprocess(images):
    images = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in images]
    images = torch.stack([transforms.ToTensor()(image) for image in images])
    return images

class ImageFloder(data.Dataset):
    def __init__(self, data, dataset, is_train=True):
        self.imagefiles = data
        self.is_train = is_train
        self.dataset = dataset

    def __getitem__(self, index):
        imageset = self.imagefiles[index]
        input_size = config[self.dataset]['input_size']
        images = [image_loader(file, input_size) for file in imageset]
        images_rgb = rgb_preprocess(images)

        return images_rgb

    def __len__(self):
        return len(self.imagefiles)
