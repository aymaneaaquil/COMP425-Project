import os
from PIL import Image
import math
import random
import shutil
from tqdm import tqdm

def rotate_and_crop(image, angle):
    # here we take the height and width, but they are the same size, so we will only be using width.
    width, height = image.size
    # now we just rotate and image and make sure the ouput contains the WHOLE rotated image, uncut.
    rotated = image.rotate(angle, resample=Image.BICUBIC, expand=True)
    # now it gets interesting, we take the angle mod90, because ater 90 ie 110, the shape is the same as 20, just with the pixels mixed up.
    angle_mod = angle % 90
    # now it gets even more interesting, so lets say the angle started off at 60, after mod90 it stays 60, but the truth is the maximum black space happens at 45deg not 90, and it start reverting after 45, so 0->45 increases the black space, and 45->90 decreases it. Ok so if we take 90 and subtract any angle less than 90 from it, we will have a corresponding angle with the same amount of black space, its equivalent small angle. so this gives us the angle of rotation from 0.
    if angle_mod > 45:
        angle_mod = 90 - angle_mod
    # here we turn to radians, important for calculation
    angle_rad = math.radians(angle_mod)
    #this is very cool, I didnt know the formula but found it here https://math.stackexchange.com/questions/828878/calculate-dimensions-of-square-inside-a-rotated-square, this gives us the smallest square that has parralel lines to the origninal borders, in a rotated square.
    crop_size = int(width / (math.cos(angle_rad) + math.sin(angle_rad)))
    # pretty easy to find center of an image
    center_x, center_y = rotated.size[0] // 2, rotated.size[1] // 2
    #then we crop based on the crop size we found
    cropped = rotated.crop((
        center_x - crop_size // 2,
        center_y - crop_size // 2,
        center_x + crop_size // 2,
        center_y + crop_size // 2
    ))
    # then we scale it back ip
    resized = cropped.resize((width, height), Image.BICUBIC)
    return resized

# so for testing this gan I really think I only need 2 split, train and test, my plan is I train on the train set, and then later I can generate on the test set to make sure it is somewhat generalizable. also notice there is no param for test_ratio, thats because it can be derived from train_ratio
def process_images(source_dir, output_dir, train_ratio=0.9, num_rotations=10):
    # we wipe dirs if they exsist
    if os.path.exists(output_dir):
        print(f"Wiping existing output directory: {output_dir}")
        shutil.rmtree(output_dir)
    # make the dirs if needed
    train_dir = os.path.join(output_dir, 'train')
    test_dir = os.path.join(output_dir, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    #the image_files are taken, from the source dir,
    image_files = [f for f in os.listdir(source_dir)]
    #then we mix them up for randomness
    random.shuffle(image_files)
    # then we find how many we want in train folder and test folder
    train_size = int(len(image_files) * train_ratio)
    train_files = image_files[:train_size]
    test_files = image_files[train_size:]
    #some print statements to make it pretty
    print(f"Processing {len(train_files)} training images...")
    #tqdm just adds a progress bar when doing stuff, makes it nice to track, not strictly necessary
    for filename in tqdm(train_files):
        process_single_image(os.path.join(source_dir, filename), train_dir, num_rotations)
    print(f"Processing {len(test_files)} testing images...")
    for filename in tqdm(test_files):
        process_single_image(os.path.join(source_dir, filename), test_dir, num_rotations)
    print(f"Augmentation complete! Generated dataset in {output_dir}")
    print(f"Train set: {len(os.listdir(train_dir))} images")
    print(f"Test set: {len(os.listdir(test_dir))} images")

# here we define how we process a single image for every image we do N rotations. 
def process_single_image(image_path, output_dir, num_rotations):
    #get image and path
    image = Image.open(image_path)
    base_filename = os.path.basename(image_path)
    #get filename with no extension
    filename_no_ext = os.path.splitext(base_filename)[0]
    image.save(os.path.join(output_dir, f"{filename_no_ext}_orig.png"))
    #for every rotation, get a random angle, get a random image for it, and save it. do as needed.
    for i in range(num_rotations):
        angle = random.randint(1, 359)
        processed = rotate_and_crop(image, angle)
        processed.save(os.path.join(output_dir, f"{filename_no_ext}_rot{i}_{angle}.png"))


#our main function for running our code
if __name__ == "__main__":
    source_directory = "data/original_data"
    data_directory = "data/processed_data"
    process_images(
        source_dir=source_directory,
        output_dir=data_directory,
        train_ratio=0.9,
        num_rotations=10
    )

