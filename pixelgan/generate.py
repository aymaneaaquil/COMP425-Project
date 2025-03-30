import os
import argparse
import glob
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image

# Import the residual generator only
from models import ResidualGenerator
from dataset import SkinLesionDataset


def load_model(model_path, device):
    """
    Load a trained generator model
    
    Args:
        model_path (str): Path to the model file
        device (torch.device): Device to load the model to
        
    Returns:
        Generator: Loaded generator model
    """
    # Always use the residual generator model
    model = ResidualGenerator(in_channels=3, out_channels=3).to(device)
    print("Using enhanced residual generator model")
    
    # Use weights_only=True to avoid security warnings
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    return model


def generate_from_dataset(args, device, generator):
    """
    Generate synthetic images from a test dataset
    
    Args:
        args (argparse.Namespace): Command line arguments
        device (torch.device): Device to run generation on
        generator (Generator): Trained generator model
    """
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Create dataset
    dataset = SkinLesionDataset(args.input_dir, transform=transform, mode='test')
    
    # Create data loader with pin_memory for faster GPU transfer
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True  # Faster data transfer to GPU
    )
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate images
    print(f"Generating {min(args.num_images, len(dataloader))} images...")
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= args.num_images:
                break
            
            # Get input image
            masked_image = batch['masked'].to(device, non_blocking=True)
            original_image = batch['original'].to(device, non_blocking=True)
            image_name = batch['name'][0]
            
            # Generate output
            generated_image = generator(masked_image)
            
            # Save results
            output_image = torch.cat((masked_image, generated_image, original_image), -2)
            save_image(output_image, os.path.join(args.output_dir, f"generated_{image_name}"),
                       normalize=True)
            
            # Save just the generated image
            save_image(generated_image, os.path.join(args.output_dir, f"gen_only_{image_name}"),
                       normalize=True)
            
            print(f"Generated image {i+1}/{min(args.num_images, len(dataloader))}")
    
    print(f"Generation complete. Results saved to {args.output_dir}")


def generate_from_raw_images(args, device, generator):
    """
    Generate synthetic images from raw input images
    
    Args:
        args (argparse.Namespace): Command line arguments
        device (torch.device): Device to run generation on
        generator (Generator): Trained generator model
    """
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get list of image files
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_files.extend(glob.glob(os.path.join(args.input_dir, ext)))
    
    # Limit the number of images
    image_files = image_files[:args.num_images]
    
    print(f"Generating {len(image_files)} images...")
    
    with torch.no_grad():
        for i, image_path in enumerate(image_files):
            # Load and preprocess image
            img = Image.open(image_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(device, non_blocking=True)
            
            # Generate output
            generated_image = generator(img_tensor)
            
            # Get filename
            filename = os.path.basename(image_path)
            
            # Save results (input and generated side by side)
            output_image = torch.cat((img_tensor, generated_image), -2)
            save_image(output_image, os.path.join(args.output_dir, f"generated_{filename}"),
                       normalize=True)
            
            # Save just the generated image
            save_image(generated_image, os.path.join(args.output_dir, f"gen_only_{filename}"),
                       normalize=True)
            
            print(f"Generated image {i+1}/{len(image_files)}")
    
    print(f"Generation complete. Results saved to {args.output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic skin lesion images')
    
    # Model and data parameters
    parser.add_argument('--model_path', type=str, help='Path to the trained generator model')
    parser.add_argument('--model_folder', type=str, help='Folder containing trained models (will use the latest generator)')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing input images')
    parser.add_argument('--output_dir', type=str, default='pixelgan_outputs/generated', help='Directory to save generated images')
    parser.add_argument('--img_size', type=int, default=256, help='Size of input/output images')
    parser.add_argument('--num_images', type=int, default=10, help='Number of images to generate')
    
    # Hardware options
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA usage')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker threads for data loading')
    
    # Generation mode
    parser.add_argument('--raw_mode', action='store_true', help='Use raw images instead of dataset')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.model_path and not args.model_folder:
        raise ValueError("Either --model_path or --model_folder must be specified")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    if device.type == 'cuda':
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        # Enable cuDNN benchmarking for performance
        torch.backends.cudnn.benchmark = True
    else:
        print("Using CPU")
    
    # Find model path
    if args.model_path:
        model_path = args.model_path
    else:
        # Find the latest generator model in the folder
        model_files = sorted(glob.glob(os.path.join(args.model_folder, "generator_*.pth")))
        if not model_files:
            raise ValueError(f"No generator model found in {args.model_folder}")
        model_path = model_files[-1]  # Use the latest one
    
    print(f"Loading model from {model_path}")
    
    generator = load_model(model_path, device)
    
    # Generate images
    if args.raw_mode:
        generate_from_raw_images(args, device, generator)
    else:
        generate_from_dataset(args, device, generator)


if __name__ == "__main__":
    main()