'''
The purpose of this is to generate a damaged dataset.
This will distort images, or removed parts of them entirely.
'''
import os
import sys
import random
if __name__ == '__main__':
    os.sys.path.append('.')
from PIL import Image
import numpy as np
import scipy
import cv2

from scripts.fs_utils import collect_images

# Deletes bottom half of rows.
def thanos_snap(source_path, target_path):
    img = Image.open(source_path, mode='r')
    pt1 = (0, img.height//2)
    pt2 = (img.width, img.height)
    img_nparray = np.asarray(img)
    cv2.rectangle(img_nparray, pt1, pt2, (128, 128, 128), -1, 8, 0)
    new_img = Image.fromarray(img_nparray.astype(np.uint8))
    new_img.save(target_path, quality=97)

def swirl(source_path, target_path):
    img = Image.open(source_path, mode='r')
    img_nparray = np.array(img)
    A = img_nparray.shape[0] / 3.0
    w = 2.0 / img_nparray.shape[1]
    shift = lambda x: A * np.sin(2.0*np.pi*x * w)
    for i in range(img_nparray.shape[0]):
        img_nparray[:,i] = np.roll(img_nparray[:,i], int(shift(i)))
    new_img = Image.fromarray(img_nparray.astype(np.uint8))
    new_img.save(target_path, quality=97)

def thanos_infinite_snap(source_path, target_path):
    img = Image.open(source_path, mode='r')
    pt1 = (0, 0)
    pt2 = (img.width, img.height)
    img_nparray = np.asarray(img)
    cv2.rectangle(img_nparray, pt1, pt2, (128, 128, 128), -1, 8, 0)
    new_img = Image.fromarray(img_nparray.astype(np.uint8))
    new_img.save(target_path, quality=97)

def main(source_dir, target_dir):
    source_images = collect_images(source_dir)
    for source_path, source_fname in source_images:
        target_path = os.path.join(target_dir, source_fname)
        decision = random.randint(0, 20)
        if decision >= 0 and decision <= 10:
            thanos_snap(source_path, target_path)
        elif decision > 10 and decision < 19:
            swirl(source_path, target_path)
        elif decision == 19:
            thanos_infinite_snap(source_path, target_path)

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage:')
        print('  python damage_images.py <source_dir> <target_dir>')
        exit()
    source_dir = sys.argv[1]
    target_dir = sys.argv[2]
    main(source_dir, target_dir)
