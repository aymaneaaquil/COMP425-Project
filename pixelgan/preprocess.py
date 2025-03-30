import os
import argparse
import random
import shutil
import numpy as np
from PIL import Image, ImageOps, ImageFilter
import cv2


def create_directory_structure(data_dir):
    """
    Creates the necessary directory structure for the dataset
    
    Args:
        data_dir (str): Base directory for the dataset
    """
    # Create main directories
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')
    
    # Create subdirectories
    for directory in [train_dir, test_dir]:
        os.makedirs(os.path.join(directory, 'originals'), exist_ok=True)
        os.makedirs(os.path.join(directory, 'masks'), exist_ok=True)


def generate_mask(image, method='edge_detection'):
    """
    Generate a mask for a skin lesion image
    
    Args:
        image (PIL.Image): Original image
        method (str): Method to use for mask generation ('edge_detection', 'thresholding', 'random')
        
    Returns:
        PIL.Image: Generated mask
    """
    # Convert PIL image to numpy array
    img_array = np.array(image)
    
    if method == 'edge_detection':
        # Convert to grayscale
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply Canny edge detection
        edges = cv2.Canny(blurred, 30, 150)
        
        # Dilate the edges to make them more visible
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=2)
        
        # Fill the enclosed regions
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(gray)
        
        # Find the largest contour (likely the skin lesion)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            cv2.drawContours(mask, [largest_contour], 0, 255, -1)
        
        # Convert back to PIL
        mask_img = Image.fromarray(mask)
        
    elif method == 'thresholding':
        # Convert to grayscale
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply Otsu's thresholding
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Convert back to PIL
        mask_img = Image.fromarray(thresh)
        
    else:  # Random method as fallback
        # Convert to grayscale
        gray_img = ImageOps.grayscale(image)
        
        # Apply a random threshold to create a simple mask
        threshold = random.randint(100, 150)
        mask_img = gray_img.point(lambda p: 255 if p > threshold else 0)
        
        # Apply blur to smooth the mask
        mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=2))
    
    # Ensure the mask is in RGB format
    mask_img = mask_img.convert('RGB')
    
    return mask_img


def split_dataset(source_dir, data_dir, test_split=0.1):
    """
    Split the dataset into training and testing sets
    
    Args:
        source_dir (str): Directory containing original images
        data_dir (str): Base directory for the dataset
        test_split (float): Proportion of data to use for testing
    """
    # Create directory structure
    create_directory_structure(data_dir)
    
    # Get list of image files
    files = [f for f in os.listdir(source_dir) 
             if os.path.isfile(os.path.join(source_dir, f)) and
             (f.endswith('.jpg') or f.endswith('.png') or f.endswith('.jpeg'))]
    
    # Shuffle files
    random.shuffle(files)
    
    # Calculate split point
    split_idx = int(len(files) * (1 - test_split))
    
    # Split files
    train_files = files[:split_idx]
    test_files = files[split_idx:]
    
    # Process training files
    for f in train_files:
        process_file(os.path.join(source_dir, f), os.path.join(data_dir, 'train'))
    
    # Process testing files
    for f in test_files:
        process_file(os.path.join(source_dir, f), os.path.join(data_dir, 'test'))


def process_file(file_path, dest_dir):
    """
    Process a single file by copying it to originals and generating a mask
    
    Args:
        file_path (str): Path to the original image file
        dest_dir (str): Destination directory (train or test)
    """
    # Get filename
    filename = os.path.basename(file_path)
    
    # Copy original file
    original_dest = os.path.join(dest_dir, 'originals', filename)
    shutil.copy(file_path, original_dest)
    
    # Generate mask
    try:
        img = Image.open(file_path).convert('RGB')
        mask = generate_mask(img)
        
        # Save mask
        mask_dest = os.path.join(dest_dir, 'masks', filename)
        mask.save(mask_dest)
    except Exception as e:
        print(f"Error processing {filename}: {str(e)}")


def main():
    parser = argparse.ArgumentParser(description='Preprocess skin lesion images')
    parser.add_argument('--source_dir', type=str, required=True, help='Directory with source images')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory to store processed data')
    parser.add_argument('--test_split', type=float, default=0.1, help='Proportion of data to use for testing')
    
    args = parser.parse_args()
    
    # Create data directory if it doesn't exist
    os.makedirs(args.data_dir, exist_ok=True)
    
    # Split and process the dataset
    split_dataset(args.source_dir, args.data_dir, args.test_split)
    
    print(f"Preprocessing complete. Data saved to {args.data_dir}")


if __name__ == "__main__":
    main()