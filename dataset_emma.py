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
        # Resize the image so that at least one dimension is equal to 128
        width, height = img.size
        aspect_ratio = width / height
        if width <= height:
            new_width = 128
            new_height = int(new_width / aspect_ratio)
        else:
            new_height = 128
            new_width = int(new_height * aspect_ratio)
        img = img.resize((new_width, new_height))
        if self.get_preprocessed_image:
            img = transforms.Pad(padding=256)(img)
            img = self.preprocess(img)
        return img
    
    def create_smaller_dataset(self, folder_name = 'smaller_dataset', nb_img=1000):
        """
        Function to create a smaller folder with images from the dataset
        
        parameters: nb_img - nb of images to be included in the new folder
        """
        new_folder = os.path.join(self.dir_image_folder, folder_name)
        if not os.path.exists(new_folder):
            os.makedirs(new_folder)
        for idx in range(nb_img):
            img = self.__getitem__(idx)
            img_path = self.image_names[idx]
            img.save(os.path.join(new_folder, os.path.basename(img_path)))
        print(f"Created folder with {nb_img} images")
        print(f"Folder path: {new_folder}")
        return new_folder
    


if __name__ == "__main__":
    #testing the dataset
    get_preprocessed_image = False
    # get_preprocessed_image = True
    # my_path = os.path.join(os.getcwd(), 'data/images/')
    my_path =  'C:/Users/emend/3A_new/3A_new/Computer Vision/h-and-m-personalized-fashion-recommendations/images/'
    dataset = myDataset(my_path, get_preprocessed_image= get_preprocessed_image)
    image_names = dataset.get_image_names(my_path)
    # print(len(dataset))
    # print(len(image_names))
    if get_preprocessed_image:
        print(dataset[20].shape)
        plt.imshow(dataset[20].permute(1, 2, 0))
        plt.show()
    else:
        print(dataset[20].size)
        plt.imshow(dataset[20])
        plt.show()
    