import os 
from torch.utils.data import Dataset
import torchvision
from PIL import Image
from torchvision import transforms
import torch
import matplotlib.pyplot as plt
 
class myDataset(Dataset):
    def __init__(self, dir_image_folder_hm, dir_image_folder_fash, get_preprocessed_image = True, dataset_type = 'both'):
        """

        Function to Initialize the Dataset

        parameters: dir_image_folder_hm(string) - Path of directory containing images of human faces
                    dir_image_folder_fash(string) - Path of directory containing images of fashion items
                    get_preprocessed_image(boolean) - If True, returns preprocessed image
                    dataset_type(string) - Type of dataset to be used. Options: 'hm', 'fash', 'both'

        """
        self.preprocess  = torchvision.models.ResNet50_Weights.IMAGENET1K_V2.transforms()
        self.dir_image_folder_hm = dir_image_folder_hm
        self.dir_image_folder_fash = dir_image_folder_fash
        self.image_names_hm = self.get_image_names_hm(dir_image_folder_hm)
        self.image_names_fash = self.get_image_names_fash(dir_image_folder_fash)
        self.image_names = self.image_names_hm + self.image_names_fash
        self.get_preprocessed_image = get_preprocessed_image
        self.dataset_type = dataset_type

    def get_image_names_fash(self, dir_image_folder_fash):
        image_names = []
        for filename in os.listdir(dir_image_folder_fash):
            fullpath = os.path.join(dir_image_folder_fash, filename)
            image_names.append(fullpath)
        return image_names

    def get_name_img(self, idx):
        return self.image_names[idx]

    def get_image_names_hm(self, dir_image_folder_hm):
        """
        Function to Combine Directory Path with individual Image Paths
        
        parameters: path(string) - Path of directory
        returns: image_names(string) - Full Image Path
        """
        image_names = []
        for dirname, _, filenames in os.walk(dir_image_folder_hm):
            for filename in filenames:
                fullpath = os.path.join(dirname, filename)
                image_names.append(fullpath)
        return image_names
    
    def __len__(self):
        if self.dataset_type == 'hm':
            return len(self.image_names_hm)
        elif self.dataset_type == 'fash':
            return len(self.image_names_fash)
        else:
            return len(self.image_names)
    
    def __getitem__(self, idx):

        if self.dataset_type == 'hm':
            image_path = self.image_names_hm[idx]
        elif self.dataset_type == 'fash':
            image_path = self.image_names_fash[idx]
        else:
            image_path = self.image_names[idx]

        img = Image.open(image_path)
        #check if the image is in the right format of 3 channels

        if self.get_preprocessed_image:
            try : 
                img = transforms.Pad(padding=256)(img)
                img = self.preprocess(img)
            except Exception as e:
                print(f"Error in preprocessing image {image_path}: {e}")
                #replace the image with a black image
                img = Image.new('RGB', (256, 256), (0, 0, 0))
                img = self.preprocess(img)
        return img

if __name__ == "__main__":
    
    get_preprocessed_image = True
    my_path_hm = os.path.join(os.getcwd(), 'data/h&mdataset/images/')
    my_path_fash = os.path.join(os.getcwd(), 'data/fashion-dataset/images/')
    print(my_path_hm)
    print(my_path_fash)
    

    #testing the dataset hm
    dataset = myDataset(my_path_hm, my_path_fash, get_preprocessed_image, 'hm')
    image_names = dataset.get_image_names_hm(my_path_hm)
    print(len(dataset))
    print(len(image_names))
    if get_preprocessed_image:
        print(dataset[20].shape)
        plt.imshow(dataset[20].permute(1, 2, 0))
        plt.show()
    else:
        print(dataset[20].size)
        plt.imshow(dataset[20])
        plt.show()
    
    #testing the dataset fash
    dataset = myDataset(my_path_hm, my_path_fash, get_preprocessed_image, 'fash')
    image_names = dataset.get_image_names_fash(my_path_fash)
    print(len(dataset))
    print(len(image_names))
    if get_preprocessed_image:
        print(dataset[20].shape)
        plt.imshow(dataset[20].permute(1, 2, 0))
        plt.show()
    else:
        print(dataset[20].size)
        plt.imshow(dataset[20])
        plt.show()
    
    #testing the dataset using both
    dataset = myDataset(my_path_hm, my_path_fash, get_preprocessed_image, 'both')
    image_names_hm = dataset.get_image_names_hm(my_path_hm)
    image_names_fash = dataset.get_image_names_fash(my_path_fash)
    print(len(dataset))
    print(len(image_names_hm))
    print(len(image_names_fash))
    if get_preprocessed_image:
        print(dataset[20].shape)
        plt.imshow(dataset[20].permute(1, 2, 0))
        plt.show()
    else:
        print(dataset[20].size)
        plt.imshow(dataset[20])
        plt.show()
    
    if get_preprocessed_image:
        print(dataset[20 + len(image_names_hm)].shape)
        plt.imshow(dataset[20+len(image_names_hm)].permute(1, 2, 0))
        plt.show()
    else:
        print(dataset[20 + len(image_names_hm)].size)
        plt.imshow(dataset[20+len(image_names_hm)])
        plt.show()

