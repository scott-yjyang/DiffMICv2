import os, torch, cv2, random
import numpy as np
from torch.utils.data import Dataset, Sampler
import torchvision.transforms as transforms
from scipy.ndimage.morphology import binary_erosion
import torchvision.transforms.functional as TF
from PIL import Image, ImageOps
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from skimage import filters
import numpy as np
import imageio
import dataloader.transforms as trans
import json, numbers
from glob import glob
import pickle

class BUDataset(Dataset):
    def __init__(self, data_list, train=True):
        self.trainsize = (224,224)
        self.train = train
        with open(data_list, "rb") as f:
            tr_dl = pickle.load(f)
        self.data_list = tr_dl

        self.size = len(self.data_list)
        #print(self.size)
        if train:
            self.transform_center = transforms.Compose([
                trans.CropCenterSquare(),
                transforms.Resize(self.trainsize),
                #trans.CenterCrop(self.trainsize),
                trans.RandomHorizontalFlip(),
                # trans.RandomVerticalFlip(),
                trans.RandomRotation(30),
                # trans.adjust_light(),
                transforms.ToTensor(),
                # lambda x: x*255
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
        else:
            self.transform_center = transforms.Compose([
                trans.CropCenterSquare(),
                transforms.Resize(self.trainsize),
                #trans.CenterCrop(self.trainsize),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                # lambda x: x*255
                ])


    def __getitem__(self, index):
        data_pac = self.data_list[index]
        img_path = data_pac['img_root']
        #cl_img, cr_img, ml_img, mr_img = None
        img = Image.open(img_path).convert('RGB')

        img_torch = self.transform_center(img)

        label = int(data_pac['label'])

        
        return img_torch, label


    def __len__(self):
        return self.size


class APTOSDataset(Dataset):
    def __init__(self, data_list, train=True):
        self.trainsize = (224,224)
        self.train = train
        with open(data_list, "rb") as f:
            tr_dl = pickle.load(f)
        self.data_list = tr_dl

        self.size = len(self.data_list)
        #print(self.size)
        if train:
            self.transform_center = transforms.Compose([
                trans.CropCenterSquare(),
                transforms.Resize(self.trainsize),
                #trans.CenterCrop(self.trainsize),
                trans.RandomHorizontalFlip(),
                trans.RandomVerticalFlip(),
                trans.RandomRotation(30),
                # transforms.GaussianBlur(3),
                #trans.adjust_light(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
        else:
            self.transform_center = transforms.Compose([
                trans.CropCenterSquare(),
                transforms.Resize(self.trainsize),
                #trans.CenterCrop(self.trainsize),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])

        #self.depths_transform = transforms.Compose([transforms.Resize((self.trainsize, self.trainsize)),transforms.ToTensor()])

    def __getitem__(self, index):
        data_pac = self.data_list[index]
        img_path = data_pac['img_root']
        #cl_img, cr_img, ml_img, mr_img = None
        img = Image.open(img_path).convert('RGB')

        img_torch = self.transform_center(img)

        label = int(data_pac['label'])

        
        return img_torch, label

    def __len__(self):
        return self.size



class ISICDataset(Dataset):
    def __init__(self, data_list, train=True):
        self.trainsize = (224,224)
        self.train = train
        with open(data_list, "rb") as f:
            tr_dl = pickle.load(f)
        self.data_list = tr_dl

        self.size = len(self.data_list)

        if train:
            self.transform_center = transforms.Compose([
                trans.CropCenterSquare(),
                transforms.Resize(self.trainsize),
                #trans.CenterCrop(self.trainsize),
                trans.RandomHorizontalFlip(),
                #trans.RandomVerticalFlip(),
                trans.RandomRotation(30),
                #trans.adjust_light(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
        else:
            self.transform_center = transforms.Compose([
                trans.CropCenterSquare(),
                transforms.Resize(self.trainsize),
                #trans.CenterCrop(self.trainsize),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
        #self.depths_transform = transforms.Compose([transforms.Resize((self.trainsize, self.trainsize)),transforms.ToTensor()])

    def __getitem__(self, index):
        data_pac = self.data_list[index]
        img_path = data_pac['img_root']
        #cl_img, cr_img, ml_img, mr_img = None
        img = Image.open(img_path).convert('RGB')

        img_torch = self.transform_center(img)

        label = int(data_pac['label'])

        
        return img_torch, label


    def __len__(self):
        return self.size


class ChestXrayDataSet(Dataset):
    def __init__(self, image_list_file, train=True):
        """
        self:
            data_dir: path to image directory.
            image_list_file: path to the file containing images
                with corresponding labels.
            transform: optional transform to be applied on a sample.
        """
        data_dir = "dataset/chest/all/images/images"
        # data_dir = "/home/yijun/project/DiffMIC/dataset/chest/all/images_enhanced"
        self.trainsize = (256, 256)
        image_names = []
        labels = []
        with open(image_list_file, "r") as f:
            for line in f:
                items = line.split()
                image_name= items[0]
                label = items[1:]
                label = [int(i) for i in label]
                label.append(1) if (np.array(label)==0).all() else label.append(0)
                image_name = os.path.join(data_dir, image_name)
                image_names.append(image_name)
                labels.append(label)

        self.image_names = image_names
        self.labels = labels
        #print(len(self.image_names))
        normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
        if train:
            self.transform_center = transforms.Compose([
                                        transforms.RandomResizedCrop(224),
                                        
                                        # transforms.RandomCrop(224),
                                        # transforms.GaussianBlur(3),
                                        trans.RandomHorizontalFlip(),
                                        trans.RandomRotation(20),
                                        transforms.ToTensor(),
                                        normalize
                                        #transforms.TenCrop(224),
                                        #transforms.Lambda
                                        #(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                                        #transforms.Lambda
                                        #(lambda crops: torch.stack([normalize(crop) for crop in crops]))
                                    ])
        else:
            # self.image_names = image_names[:1000]
            # self.labels = labels[:1000]
            self.transform_center = transforms.Compose([
                                        transforms.Resize(224),
                                        # transforms.TenCrop(224),
                                        # transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        normalize
                                        # transforms.Lambda
                                        # (lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                                        # transforms.Lambda
                                        # (lambda crops: torch.stack([transforms.RandomHorizontalFlip()(crop) for crop in crops])),
                                        # # transforms.Lambda
                                        # # (lambda crops: torch.stack([transforms.RandomRotation(20)(crop) for crop in crops])),
                                        # transforms.Lambda
                                        # (lambda crops: torch.stack([normalize(crop) for crop in crops]))
                                    ])

    def __getitem__(self, index):
        """
        self:
            index: the index of item
        Returns:
            image and its labels
        """
        image_name = self.image_names[index]
        image = Image.open(image_name).convert('RGB')
        label = self.labels[index]
        image = self.transform_center(image)
        return image, torch.FloatTensor(label)

    def __len__(self):
        return len(self.image_names)
