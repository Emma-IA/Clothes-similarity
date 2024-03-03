import os 
from torch.utils.data import Dataset
import torchvision
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

class myDataset(Dataset):
    def __init__(self, dir_image_folder, get_preprocessed_image = True):
        self.preprocess  = torchvision.models.ResNet50_Weights.IMAGENET1K_V2.transforms()
        self.dir_image_folder = dir_image_folder
        self.image_names = self.get_image_names(dir_image_folder)
        self.get_preprocessed_image = get_preprocessed_image

    def get_image_names(self, dir_image_folder):
        """
        Function to Combine Directory Path with individual Image Paths
        
        parameters: path(string) - Path of directory
        returns: image_names(string) - Full Image Path
        """
        image_names = []
        for dirname, _, filenames in os.walk(dir_image_folder):
            for filename in filenames:
                fullpath = os.path.join(dirname, filename)
                image_names.append(fullpath)
        return image_names
    
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, idx):
        image_path = self.image_names[idx]
        img = Image.open(image_path)
        if self.get_preprocessed_image:
            img = transforms.Pad(padding=256)(img)
            img = self.preprocess(img)
        return img

if __name__ == "__main__":
    #testing the dataset
    get_preprocessed_image = False
    my_path = os.path.join(os.getcwd(), 'data/images/')
    print(my_path)
    dataset = myDataset(my_path, get_preprocessed_image= get_preprocessed_image)
    image_names = dataset.get_image_names(my_path)
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