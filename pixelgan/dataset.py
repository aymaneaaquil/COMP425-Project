import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class SkinLesionDataset(Dataset):
    """
    Dataset class for skin lesion image pairs (original and masked)
    """
    def __init__(self, root_dir, transform=None, mode='train', masked_dir=None):
        """
        Args:
            root_dir (str): Directory with all the original images
            transform (callable, optional): Optional transform to be applied on a sample
            mode (str): 'train' or 'test'
            masked_dir (str, optional): Directory with the masked images. If None, 
                                        assumes subdirectories 'originals' and 'masks' in root_dir
        """
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode
        
        if masked_dir is None:
            # Assume directory structure as root_dir/originals and root_dir/masks
            self.original_dir = os.path.join(root_dir, 'originals')
            self.masked_dir = os.path.join(root_dir, 'masks')
        else:
            self.original_dir = root_dir
            self.masked_dir = masked_dir
        
        # Get list of image files
        self.files = [f for f in os.listdir(self.original_dir) 
                      if os.path.isfile(os.path.join(self.original_dir, f)) and
                      (f.endswith('.jpg') or f.endswith('.png') or f.endswith('.jpeg'))]
        
        # Pre-load cache paths for faster access
        self.original_paths = [os.path.join(self.original_dir, f) for f in self.files]
        self.masked_paths = [os.path.join(self.masked_dir, f) for f in self.files]
        
        print(f"Found {len(self.files)} images in {self.original_dir}")
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_name = self.files[idx]
        original_path = self.original_paths[idx]
        masked_path = self.masked_paths[idx]
        
        # Load images
        original_image = Image.open(original_path).convert('RGB')
        
        # Try to load the masked image, if it doesn't exist, create a simple mask
        try:
            masked_image = Image.open(masked_path).convert('RGB')
        except (FileNotFoundError, IOError):
            # Create a simple greyscale version as a placeholder
            masked_image = original_image.convert('L').convert('RGB')
        
        # Apply transformations if available
        if self.transform:
            original_image = self.transform(original_image)
            masked_image = self.transform(masked_image)
        
        # During testing, we might want to keep track of the image names
        if self.mode == 'test':
            return {'original': original_image, 'masked': masked_image, 'name': img_name}
        
        return {'original': original_image, 'masked': masked_image}


def get_data_loaders(data_dir, batch_size=8, img_size=256, num_workers=4):
    """
    Creates training and testing data loaders optimized for GPU
    
    Args:
        data_dir (str): Base directory with image data
        batch_size (int): Batch size
        img_size (int): Size to which images should be resized
        num_workers (int): Number of worker threads for data loading
        
    Returns:
        train_loader, test_loader (DataLoader): Training and testing data loaders
    """
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),  # Data augmentation
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Test transformations (no augmentation)
    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Create datasets
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')
    
    train_dataset = SkinLesionDataset(train_dir, transform=transform, mode='train')
    test_dataset = SkinLesionDataset(test_dir, transform=test_transform, mode='test')
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,  # Faster data transfer to GPU
        drop_last=True,   # Make sure all batches have same size
        persistent_workers=True if num_workers > 0 else False  # Keep workers alive between epochs
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,  # Use batch size of 1 for testing
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True  # Faster data transfer to GPU
    )
    
    return train_loader, test_loader