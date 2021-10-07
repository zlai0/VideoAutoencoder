import argparse
import numpy as np
import copy
from evo.core import metrics
from evo.tools import file_interface
from evo.core import sync
from evo.tools import plot
from evo.main_ape import ape
from glob import glob
from pprint import pprint
import matplotlib.pyplot as plt
from tqdm import tqdm
import ipdb

def eval_pose(ref_file, est_file):
    traj_ref = file_interface.read_kitti_poses_file(ref_file)
    traj_est = file_interface.read_kitti_poses_file(est_file)

    pose_relation = metrics.PoseRelation.translation_part
    results = ape(traj_ref, traj_est, pose_relation, align=True, correct_scale=True)
    return results.stats

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('pose_dir', type=str)
    args = parser.parse_args()

    pose_dir = args.pose_dir

    values_rmse, values_mean, values_max = [], [], []

    max_i = len(glob(f'{pose_dir}/video_*_true.txt'))
    print(f'Evaluating {max_i} trajectories.')

    for i in tqdm(range(max_i)):
        est_file = f'{pose_dir}/video_{i}_pred.txt'
        ref_file = f'{pose_dir}/video_{i}_true.txt'
        results = eval_pose(ref_file, est_file)
        values_rmse.append(results['rmse'])
        values_mean.append(results['mean'])
        values_max.append(results['max'])

    avg_rmse = np.mean(np.array(values_rmse))
    avg_mean = np.mean(np.array(values_mean))
    avg_max = np.mean(np.array(values_max))

    pprint({'avg_rmse':avg_rmse, 'avg_mean':avg_mean, 'avg_max':avg_max})