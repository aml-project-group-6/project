'''
Samples a random subset of the preprocessed images.
'''
import os
import sys
import random
import shutil
from scripts.fs_utils import collect_images

def ensure_that_eyes_come_in_pairs(source_images):
    for i in range(0, len(source_images), 2):
        s0 = source_images[i][0].split('/')[-1].split('_')[0]
        s1 = source_images[i+1][0].split('/')[-1].split('_')[0]
        if s0 != s1:
            print('Oops')
            exit()

def main(source_dir, target_dir, sample_size):
    source_images = collect_images(source_dir)
    if not source_images:
        print('Nothing to sample from.')
        exit()
    source_images.sort(key=lambda x: x[0])
    ensure_that_eyes_come_in_pairs(source_images)
    n = len(source_images)
    assert n % 2 == 0
    for _ in range(sample_size):
        i = random.randint(0, n//2)
        source_path0 = source_images[2*i][0]
        source_path1 = source_images[2*i + 1][0]
        source_fname0 = source_images[2*i][1]
        source_fname1 = source_images[2*i + 1][1]
        target_path0 = os.path.join(target_dir, source_fname0)
        target_path1 = os.path.join(target_dir, source_fname1)
        shutil.copyfile(source_path0, target_path0)
        shutil.copyfile(source_path1, target_path1)

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print('Usage:')
        print('  python sample_random_preprocessed.py <source_dir> <target_dir> <sample_size>')
        exit()
    source_dir = sys.argv[1]
    target_dir = sys.argv[2]
    sample_size = int(sys.argv[3])
    main(source_dir, target_dir, sample_size)
