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


class CUB200FewShot(Dataset):
    """
    CUB-200 数据集，支持真正的 Few-Shot 划分
    
    标准划分：
    - Base classes (训练): 类别 1-100
    - Novel classes (测试): 类别 101-200
    
    这样测试时模型面对的是从未见过的类别，才是真正的 few-shot 评估
    """
    def __init__(self, root_dir, split='base', transform=None, 
                 base_classes=100, use_all_images=True):
        """
        Args:
            root_dir: Path to CUB_200_2011 directory
            split: 'base' (类别1-100) or 'novel' (类别101-200) or 'all'
            transform: Image transforms
            base_classes: Number of base classes (default 100)
            use_all_images: If True, use both train and test images for the split
        """
        self.root_dir = root_dir
        self.transform = transform
        self.split = split
        self.base_classes = base_classes
        
        # Paths
        self.images_dir = os.path.join(root_dir, 'images')
        images_txt = os.path.join(root_dir, 'images.txt')
        split_txt = os.path.join(root_dir, 'train_test_split.txt')
        
        # Read metadata
        self.images_df = pd.read_csv(images_txt, sep=' ', names=['image_id', 'image_name'])
        self.split_df = pd.read_csv(split_txt, sep=' ', names=['image_id', 'is_training'])
        
        # Merge
        self.data = pd.merge(self.images_df, self.split_df, on='image_id')
        
        # Extract class ID from image name (e.g., "001.Black_footed_Albatross/..." -> 0)
        self.data['class_id'] = self.data['image_name'].apply(
            lambda x: int(x.split('.')[0]) - 1  # 0-indexed
        )
        
        # Filter by split
        if split == 'base':
            # 类别 0-99 (原始 1-100)
            self.data = self.data[self.data['class_id'] < base_classes]
        elif split == 'novel':
            # 类别 100-199 (原始 101-200)
            self.data = self.data[self.data['class_id'] >= base_classes]
        # else 'all': use all classes
        
        # Optionally filter by train/test split within the class split
        if not use_all_images:
            if split == 'base':
                # For base classes, use training images
                self.data = self.data[self.data['is_training'] == 1]
            else:
                # For novel classes, use test images
                self.data = self.data[self.data['is_training'] == 0]
        
        self.image_paths = self.data['image_name'].tolist()
        self.labels = self.data['class_id'].tolist()
        
        # Remap labels to be contiguous within split
        unique_labels = sorted(set(self.labels))
        self.label_map = {old: new for new, old in enumerate(unique_labels)}
        self.labels = [self.label_map[l] for l in self.labels]
        self.num_classes = len(unique_labels)
        
        print(f"CUB200FewShot: split={split}, classes={self.num_classes}, images={len(self)}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_name = self.image_paths[idx]
        img_path = os.path.join(self.images_dir, img_name)
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            image = Image.new('RGB', (224, 224))
            
        if self.transform:
            image = self.transform(image)
            
        return image, label


def get_cub_transforms(image_size=224, augment=False):
    """
    Get transforms for CUB dataset
    
    Args:
        image_size: Target image size
        augment: If True, apply data augmentation (for training)
    """
    if augment:
        return transforms.Compose([
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


def get_cub_inverse_transform():
    """Helper to un-normalize for visualization"""
    return transforms.Compose([
        transforms.Normalize(mean=[0., 0., 0.], std=[1/0.229, 1/0.224, 1/0.225]),
        transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.]),
    ])
