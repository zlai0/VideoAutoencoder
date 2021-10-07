"""
Generate Matterport3D videos for pretraining.
You need to clone the Synsin repository: https://github.com/facebookresearch/synsin and place this file there.
You also need the habitat simulator as described in the README.
"""

from data.habitat_data import *
from options.train_options import ArgumentParser
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import signal

import habitat
import shutil
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.utils.visualizations import maps
from habitat.utils.visualizations.utils import images_to_video

from tqdm import tqdm

def generate_video_from_path(path, outpath, scenename):
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    argString = '--dataset mp3d'

    opt, _ = ArgumentParser().parse(argString)

    opt.train_data_path = (path)
    opt.scenes_dir = "/home/zihang/developer/data/matterport/" # this should store mp3d
    opt.config = "./configs/mp3d.yaml" # this should store mp3d configs
    opt.num_views = 30
    opt.image_type = "zihang"
    opt.W = 256

    data = HabitatImageGenerator('train', opt, vectorize=False)

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

    iter_data_loader = iter(dataloader)
    batch = next(iter_data_loader)


    env = data.image_generator.env
    env_sim = data.image_generator.env_sim
    config = data.image_generator.config
    #     mode = "greedy"
    mode = "geodesic_path"



    IMAGE_DIR = outpath
    if not os.path.exists(IMAGE_DIR):
        os.makedirs(IMAGE_DIR)

    ###
    goal_radius = env.episodes[0].goals[0].radius
    if goal_radius is None:
        goal_radius = config.SIMULATOR.FORWARD_STEP_SIZE
    follower = ShortestPathFollower(env_sim, goal_radius, False)
    follower.mode = mode

    print("Environment creation successful")
    for episode in range(len(env.episodes)):
        # Start the timer. Once 30 seconds are over, a SIGALRM signal is sent.
        signal.alarm(15)
        # This try/except loop ensures that
        #   you'll catch TimeoutException when it's sent.
        try:
            env.reset()
            print(env._current_episode_index)
            dirname = os.path.join(
                outpath, scenename + "_%04d" % episode
            )
            if os.path.exists(dirname):
                shutil.rmtree(dirname)
            os.makedirs(dirname)
            print("Agent stepping around inside environment.")
            images = []
            while not env.episode_over:
                best_action = follower.get_next_action(
                    env.current_episode.goals[0].position
                )
                observations = env.step(best_action.value)
                im = observations["rgb"]
                output_im = im
                images.append(output_im)
            try:
                images_to_video(images, dirname, "trajectory_train")
            except:
                print("failed to save")

            print("Episode finished")
        except TimeoutException:
            print(env._current_episode_index, 'time exceeds')
            continue # continue the for loop if function A takes more than 5 second
        else:
            # Reset the alarm
            signal.alarm(0)

    env.close()
    env_sim.close()


######
class TimeoutException(Exception):   # Custom exception class
    pass

def timeout_handler(signum, frame):   # Custom signal handler
    raise TimeoutException

# Change the behavior of SIGALRM
signal.signal(signal.SIGALRM, timeout_handler)
######


datapath = '/home/zihang/developer/data/matterport/pointnav/mp3d/train/content_mini/'
outpath = '/home/zihang/developer/data/matterport/videos/train/'
listdir = sorted(os.listdir(datapath))

for fp_i, fp in enumerate(listdir):
    print(fp_i, 'NOW PROCESSING:', fp)
    # 33 and 43 somehow fails, maybe they could work for you?
    if fp_i in [33, 43]:
        continue
    scenename = fp.split('.')[0]
    generate_video_from_path(datapath + fp, outpath, scenename)