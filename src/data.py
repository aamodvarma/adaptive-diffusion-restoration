import os
import random
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

class CleanFFHQDataset(Dataset):
    def __init__(self, image_dir, filenames, transform=None):
        self.image_dir = image_dir
        self.filenames = filenames
        self.transform = transform
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.filenames[idx])
        try:
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img
        except Exception as e:
            # Handle corrupted images by returning a random valid image
            print(f"Error loading image {img_path}: {e}")
            return self.__getitem__(random.randint(0, len(self.filenames) - 1))

def get_clean_ffhq_dataloaders(image_dir, train_ratio=0.8, batch_size=64, seed=123, img_size=256):
    '''
    Splits clean FFHQ images into train/test and returns dataloaders (no masking applied).
    '''
    rng = random.Random(seed)
    clean_filenames = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
    
    # Filter out non-image files
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    clean_filenames = [f for f in clean_filenames if os.path.splitext(f.lower())[1] in valid_extensions]
    
    rng.shuffle(clean_filenames)
    split_idx = int(len(clean_filenames) * train_ratio)
    train_files, test_files = clean_filenames[:split_idx], clean_filenames[split_idx:]
    
    # Training transform with augmentation
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(p=0.5),  # Data augmentation
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),  # Color augmentation
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])
    
    # Test transform without augmentation
    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = CleanFFHQDataset(image_dir, train_files, train_transform)
    test_dataset = CleanFFHQDataset(image_dir, test_files, test_transform)
    
    # Reduced batch size and added pin_memory for faster data loading
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4,  # Reduced from 8
        pin_memory=True,
        drop_last=True  # Ensures consistent batch sizes
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Training samples: {len(train_files)}")
    print(f"Test samples: {len(test_files)}")
    
    return train_loader, test_loader


# Alternative: If you want to keep unnormalized data for MAE
def get_clean_ffhq_dataloaders_unnormalized(image_dir, train_ratio=0.8, batch_size=64, seed=123):
    '''
    Version without normalization - sometimes better for MAE
    '''
    rng = random.Random(seed)
    clean_filenames = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
    
    # Filter out non-image files
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    clean_filenames = [f for f in clean_filenames if os.path.splitext(f.lower())[1] in valid_extensions]
    
    rng.shuffle(clean_filenames)
    split_idx = int(len(clean_filenames) * train_ratio)
    train_files, test_files = clean_filenames[:split_idx], clean_filenames[split_idx:]
    
    # Training transform with light augmentation
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),  # Values in [0, 1]
    ])
    
    # Test transform
    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    train_dataset = CleanFFHQDataset(image_dir, train_files, train_transform)
    test_dataset = CleanFFHQDataset(image_dir, test_files, test_transform)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Training samples: {len(train_files)}")
    print(f"Test samples: {len(test_files)}")
    
    return train_loader, test_loader
