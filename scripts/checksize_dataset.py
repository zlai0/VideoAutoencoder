# check dataset size.
# usage: python sanity_check_dataset.py 'path-to-dataset'
import glob
import sys

data_dir = sys.argv[1]
files = glob.glob(f'{data_dir}/*/*.jpg')
print('Dataset total images:', len(files))