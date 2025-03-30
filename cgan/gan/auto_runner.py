import subprocess
import datetime
import sys

# ok so below I set a bunch of "experiments" these are ways I want to train the model, so pretty much just try different things out, and see what works, as a baseline I took what the model started with, which can be found here: https://github.com/eriklindernoren/PyTorch-GAN/tree/master/implementations/ccgan, here are the default params by the way python cgan.py --n_epochs 200 --batch_size 8 --dataset_name "img_align_celeba" --lr 0.0002 --b1 0.5 --b2 0.999 --n_cpu 8 --latent_dim 100 --img_size 128 --mask_size 32 --channels 3 --sample_interval 500, I decided to only focus on n_epochs, batch_size, I set image size in although it doesnt even need to be set, then lr, and mask size. with a few changes here and there, I also change the n epochs to 50 for most because it would take a while to train longer, then I messed around with the learning rate, and batch size, and mask size, nad latent dimensions, and also b1 and b2 which are hyper params for adam for first and second moment estimates, b1 decides how much of past gradient is maintained and b2 helps scale the learning rate, I dont fully understand them so didnt mess around with them too much.

EXPERIMENTS = [

    "python gan/cgan.py --n_epochs 1 --batch_size 8 --img_size 128 --lr 0.0002 --mask_size 32", #test one

    "python gan/cgan.py --n_epochs 2 --batch_size 8 --img_size 128 --lr 0.0002 --mask_size 32", #test two

    "python gan/cgan.py --n_epochs 100 --batch_size 4 --img_size 128 --lr 0.0002 --mask_size 32", #220325-170511

    "python gan/cgan.py --n_epochs 50 --batch_size 8 --img_size 128 --lr 0.0001 --mask_size 32",  #230325-002603

    "python gan/cgan.py --n_epochs 50 --batch_size 8 --img_size 128 --lr 0.0005 --mask_size 32",  #230325-035402

    "python gan/cgan.py --n_epochs 50 --batch_size 8 --img_size 128 --lr 0.0002 --b1 0.7 --b2 0.999 --mask_size 32", #230325-072221

    "python gan/cgan.py --n_epochs 50 --batch_size 8 --img_size 128 --lr 0.0002 --mask_size 48",  #230325-104109

    "python gan/cgan.py --n_epochs 50 --batch_size 8 --img_size 128 --lr 0.0002 --mask_size 24",  #230325-140100

    "python gan/cgan.py --n_epochs 50 --batch_size 8 --img_size 128 --lr 0.0002 --mask_size 32",  #230325-172325
]

# I forgot to mention above but the purpose of all this is that its a grind to every 4 hours click run, why not have it all queued up? that is what this script does.
def run_experiments():
    print(f"Total experiments to run: {len(EXPERIMENTS)}")

    # here i print some info and run the commands, the datetime is pretty cool, strftime formats as a string
    for i, command in enumerate(EXPERIMENTS):
        print(f"\n{'=' * 50}")
        print(f"Running experiment {i+1}/{len(EXPERIMENTS)}")
        print(f"Command: {command}")
        print(f"Start time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'-' * 50}")

        #here is the meat, we run the code in my shell, check=true just means it will raise an exception if the command fails. after we have some other exceptions, to make it cleaner when i do ctrl+c to exit.
        try:
            process = subprocess.run(command, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Experiment failed with error code {e.returncode}")
        except KeyboardInterrupt:
            print("\nExperiment interrupted by user")
            sys.exit(1)

        #more print statements when we finish
        print(f"{'-' * 50}")
        print(f"Finished experiment {i+1}/{len(EXPERIMENTS)}")
        print(f"End time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'=' * 50}")

    print(f"\nAll experiments completed at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    # here I can just run the code, and if i ctrl+c out, it wont crashout on me, and will cleanly exit
    try:
        run_experiments()
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
        sys.exit(1)