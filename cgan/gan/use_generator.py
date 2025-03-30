import argparse
import os
import torch
import numpy as np
import random
from torchvision.utils import save_image
from PIL import Image
import torchvision.transforms as transforms
from models import Generator
import glob
import shutil

# this was interesting, this is quite similar to the cgan.py, with argparse, but now I uses it to make a script that uses the traing generator
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="gan/models/230325-172325/generator_50.pth")
parser.add_argument("--data_dir", type=str, default="data/processed_data/train")
parser.add_argument("--output_dir", type=str, default="gan/generated_images")
parser.add_argument("--num_images", type=int, default=282)
parser.add_argument("--img_size", type=int, default=128)
parser.add_argument("--num_masks", type=int, default=1)
parser.add_argument("--mask_size", type=int, default=32)
parser.add_argument("--mask_combos", type=int, default=32)
parser.add_argument("--channels", type=int, default=3)
opt = parser.parse_args()

# so we want to output all the images to a folder called generated images, not we are switching our approach a bit from the cgan where we had one folder per model. maybe I could have done it like in cgan, anyways. We wipe the dir if it exsists
if os.path.exists(opt.output_dir):
    print(f"Clearing existing output directory: {opt.output_dir}")
    shutil.rmtree(opt.output_dir)

#here we made the other dirs, the subdirs, one called actual_images, where we hold just the generated images, so lets say my friend Aymane asks for some images to train a model, I can just pull from this subdir, no biggie. we also have one called comapred_images, that just to give some reference, we have a 3 stack of images, top is the one with the gap, middle is the generated one, bottom is the downscaled image we also provided.
os.makedirs(opt.output_dir, exist_ok=True)
actual_images_dir = os.path.join(opt.output_dir, "actual_images")
compared_images_dir = os.path.join(opt.output_dir, "compared_images")
os.makedirs(actual_images_dir, exist_ok=True)
os.makedirs(compared_images_dir, exist_ok=True)

# if we have cuda we can use it.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# now we setup the generator, quite similarly to our cgan, we load our generator.
input_shape = (opt.channels, opt.img_size, opt.img_size)
generator = Generator(input_shape).to(device)

# here we load the model we saved earlier, we set it to eval mode rather than train mode, so things like dropout are disabled.
generator.load_state_dict(torch.load(opt.model_path, map_location=device))
generator.eval()

# we use the same transforms as in the cgan
transform_hr = transforms.Compose([
    transforms.Resize((opt.img_size, opt.img_size), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

transform_lr = transforms.Compose([
    transforms.Resize((opt.img_size // 4, opt.img_size // 4), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# now we get some random images, as many as specified by our args.
def get_random_images(data_dir, num_images):
    image_files = glob.glob(os.path.join(data_dir, '**', '*.png'), recursive=True)
    if not image_files:
        raise Exception(f"No image files found in {data_dir}")
    selected_files = random.sample(image_files, min(num_images, len(image_files)))
    return selected_files

# this is a bit different, I decided to switch something up compared to our cgan. so in our cgan we apply one mask, but I was thinking, wouldnt it be cool to see what the model could do with multiple masks applied? Anyways that is what I did.
# let me show you a quick step by step of how we calculate, first keep in mind the origin is top left corner., then lets say we pick 2 random numbers, 50 and 20. those are x1y1, then we get x2y2 by adding the mask_size, if mask is 32 then we would get x2y2 = 82,52. so x1y1 is the top left corner of the mask, and x2y2 is the bottom right corner. thena fter we have that we set those coords to -1 which is black. and we do this for every mask we apply.
def apply_multiple_masks(img, num_masks, mask_size):
    masked_img = img.clone()
    for _ in range(num_masks):
        y1 = random.randint(0, opt.img_size - mask_size - 1)
        x1 = random.randint(0, opt.img_size - mask_size - 1)
        y2, x2 = y1 + mask_size, x1 + mask_size
        masked_img[:, y1:y2, x1:x2] = -1
    return masked_img


# this will be our main execution here, we get the image files, remove the extension, load the image, add batch dimension, then we do a loop for every output image we want, see comments below.

print(f"Processing {opt.num_images} images with {opt.mask_combos} mask combinations each...")
image_files = get_random_images(opt.data_dir, opt.num_images)
print(f"Selected {len(image_files)} random images")

for i, image_path in enumerate(image_files):
    filename = os.path.splitext(os.path.basename(image_path))[0]
    img = Image.open(image_path).convert('RGB')
    hr_img = transform_hr(img)
    lr_img = transform_lr(img)
    
    # here we need to add a batch dimension, because our model expects one, so we add a batch of 1, so ie 3,128,128 becomes 1,3,128,128.
    hr_img = hr_img.unsqueeze(0).to(device)
    lr_img = lr_img.unsqueeze(0).to(device)
    
    # Tnow here we do a loop so for every mask combo, this is like the number of output images we want per image, ie 1->10. we get a masked image. then we plug in the masked image with the downscaled one into the generator, like in the cgan.
    for combo in range(opt.mask_combos):
        masked_img = apply_multiple_masks(hr_img[0], opt.num_masks, opt.mask_size)
        #we need the batch dimension again
        masked_img = masked_img.unsqueeze(0)
        
        with torch.no_grad():
            gen_img = generator(masked_img, lr_img)
        
        # Now we save our generated files, we save a comparison using torch.cat which stacks the images. on the -2 dimension which is heigh so vertical stack. 
        comparison = torch.cat((masked_img, gen_img, hr_img), -2)
        #we save our comparison image.
        save_image(comparison, f"{compared_images_dir}/{filename}_combo{combo+1}_comparison.png", normalize=True)
        
        # now we save our generated image
        save_image(gen_img, f"{actual_images_dir}/{filename}_combo{combo+1}_inpainted.png", normalize=True)
        
        #below we have some print statements just to follow what is happening
        print(f"Processed image {i+1}/{len(image_files)}: {filename}, mask combination {combo+1}/{opt.mask_combos}")
    
    print(f"  Applied {opt.num_masks} masks of size {opt.mask_size}x{opt.mask_size} in {opt.mask_combos} different combinations")

print(f"All images processed and saved to {opt.output_dir}")
print(f"- Inpainted images saved to: {actual_images_dir}")
print(f"- Comparison images saved to: {compared_images_dir}")