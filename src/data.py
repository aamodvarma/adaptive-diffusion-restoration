from torch.utils.data import Dataset, DataLoader
import os
import random
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
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img

def get_clean_ffhq_dataloaders(image_dir, train_ratio=0.8, batch_size=64, seed=123):
    '''
    Splits clean FFHQ images into train/test and returns dataloaders (no masking applied).
    '''
    rng = random.Random(seed)
    clean_filenames = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
    random.shuffle(clean_filenames)

    split_idx = int(len(clean_filenames) * train_ratio)
    train_files, test_files = clean_filenames[:split_idx], clean_filenames[split_idx:]

    transform = transforms.Compose([transforms.ToTensor(),])

    train_dataset = CleanFFHQDataset(image_dir, train_files, transform)
    test_dataset = CleanFFHQDataset(image_dir, test_files, transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader
