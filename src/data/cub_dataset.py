import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class CUB200Dataset(Dataset):
    def __init__(self, root_dir, train=True, transform=None):
        """
        Args:
            root_dir (string): Path to the CUB_200_2011 directory.
            train (bool): If True, load training set, else load test set.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.train = train
        
        # Paths
        self.images_dir = os.path.join(root_dir, 'images')
        images_txt = os.path.join(root_dir, 'images.txt')
        split_txt = os.path.join(root_dir, 'train_test_split.txt')
        
        # Read metadata
        # images.txt: <image_id> <image_name>
        self.images_df = pd.read_csv(images_txt, sep=' ', names=['image_id', 'image_name'])
        
        # train_test_split.txt: <image_id> <is_training_image>
        self.split_df = pd.read_csv(split_txt, sep=' ', names=['image_id', 'is_training'])
        
        # Merge and filter
        self.data = pd.merge(self.images_df, self.split_df, on='image_id')
        
        if self.train:
            self.data = self.data[self.data['is_training'] == 1]
        else:
            self.data = self.data[self.data['is_training'] == 0]
            
        self.image_paths = self.data['image_name'].tolist()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_name = self.image_paths[idx]
        img_path = os.path.join(self.images_dir, img_name)
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image in case of error to not crash training
            image = Image.new('RGB', (224, 224))
            
        if self.transform:
            image = self.transform(image)
            
        return image

def get_cub_transforms(image_size=224):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        # Normalize with ImageNet stats
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def get_cub_inverse_transform():
    # Helper to un-normalize for visualization
    return transforms.Compose([
        transforms.Normalize(mean=[0., 0., 0.], std=[1/0.229, 1/0.224, 1/0.225]),
        transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.]),
    ])
