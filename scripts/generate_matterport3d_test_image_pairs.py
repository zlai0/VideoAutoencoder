"""
Generate Matterport3D image pairs (in the same way as synsin) for testing.
You need to clone the Synsin repository: https://github.com/facebookresearch/synsin and place this file there.
You also need the habitat simulator as described in the README.
"""

from data.habitat_data import *
from options.train_options import ArgumentParser
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

argString = '--dataset replica --images_before_reset 200'

opt, _ = ArgumentParser().parse(argString)

opt.test_data_path = (
    "data/scene_episodes/replica_test/dataset_one_ep_per_scene.json.gz"
)

opt.scenes_dir = "/home/zihang/developer/data/Replica-Dataset/replica_v1/"
opt.config = "./configs/mp3d.yaml"
opt.W = 256

data = HabitatImageGenerator('test', opt, vectorize=False)

def worker_init_fn(worker_id):
    torch.manual_seed(worker_id)

dataloader = DataLoader(
    data,
    shuffle=False,
    drop_last=False,
    batch_size=1,
    num_workers=0,
    pin_memory=True,
    worker_init_fn=worker_init_fn,
)

dataset_output_path = '/home/zihang/developer/data/replica_test/'
os.makedirs(dataset_output_path, exist_ok=True)

images_before_reset = 200
N_iter = images_before_reset * 5
iter_data_loader = iter(dataloader)
for i in tqdm(range(N_iter-1)):
    batch = next(iter_data_loader)
    im0 = batch['images'][0][0].permute(1,2,0)
    im1 = batch['images'][1][0].permute(1,2,0)

    save_path = dataset_output_path + str(i).zfill(6)
    os.makedirs(save_path, exist_ok=True)
    plt.imsave(save_path+'/im0.png', im0.numpy())
    plt.imsave(save_path+'/im1.png', im1.numpy())