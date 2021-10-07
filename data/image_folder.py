import yaml
from glob import glob
import numpy as np

with open('configs/datasets.yaml') as f:
    config_data = f.read()
    config = yaml.safe_load(config_data)

IMG_EXT = ['png', 'jpg']

def dataloader(dataset, clip_length, interval, n_valid=20, is_train=True, load_all_frames=False):
    if is_train:
        data_path = config[dataset]['train_path']
    else:
        data_path = config[dataset]['test_path']

    if data_path is None:
        raise RuntimeError(f"dataset does not support {'train' if is_train else 'test'} mode.")

    video_paths = sorted(glob(data_path+'/*/'))
    batches_train, batches_valid = [], []
    for index in range(len(video_paths)):
        vpath = video_paths[index]
        fnames = sum([sorted(glob(vpath+f'/*.{ext}')) for ext in IMG_EXT],[])
        fnames = fnames[::interval]

        video_batches = []
        if load_all_frames:
            video_batches.append(fnames)
        else:
            while True:
                if len(fnames) < clip_length:
                    break

                frame_sequence = fnames[:clip_length]
                video_batches.append(frame_sequence)
                fnames = fnames[1:]  # skip first one

        if index >= n_valid:
            batches_train.extend(video_batches)
        else:
            batches_valid.extend(video_batches)

    return batches_train, batches_valid
