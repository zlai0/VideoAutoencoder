from multiprocessing import Pool
from glob import glob
from tqdm import tqdm
import subprocess
import os

def move_folder(folder):
    files = sorted(glob(f'{folder}/*.jpg'))
    files_to_transfer = files[::3] # Reduce temporal resolution to 1/3
    for file in files_to_transfer:
        target = file.replace('dataset', 'subsampled_dataset')
        cmd1 = f'mkdir -p {os.path.dirname(target)}'
        cmd2 = f'cp {file} {target}'
        subprocess.call(cmd1, shell=True)
        subprocess.call(cmd2, shell=True)

if __name__ == '__main__':
    folders = glob('./dataset/train/*')
    target_dir = './subsampled_dataset/train/'

    with Pool(24) as p:
        r = list(tqdm(p.imap(move_folder, folders), total=len(folders)))