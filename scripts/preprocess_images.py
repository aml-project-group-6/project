import os
import sys
import multiprocessing.pool

from PIL import Image
import numpy as np
import cv2
if __name__ == '__main__':
    os.sys.path.append('.')
from scripts.fs_utils import collect_images, ensure_directory_exists

# The amount of concurrency available. This should not be set
# higher than the number of hardware threads available on the
# machine, minus 1.
THREAD_COUNT = 5
# The amount of images in a single batch. We divide the images
# into batches and assign those batches to different threads.
BATCH_SIZE = 500
# The expected size of the processed images at the end.
CROP_SIZE = 512

def convert(filename, crop_size):
    image = Image.open(filename, mode='r')
    assert len(np.shape(image)) == 3, 'Shape of image {} unexpected'.format(filename)
    width, height = image.size
    converted = None
    if width / float(height) >= 1.3:
        cols_thres = np.where(np.max(np.max(np.asarray(image), axis=2), axis=0) > 35)[0]
        if len(cols_thres) > crop_size//2:
            min_x, max_x = cols_thres[0], cols_thres[-1]
        else:
            min_x, max_x = 0, -1
        converted = image.crop((min_x, 0, max_x, height))
    else:
        converted = image
    converted = converted.resize((crop_size, crop_size), resample=Image.BILINEAR)
    enhanced_image = enhance_contrast(np.asarray(converted), radius=crop_size//2)
    return Image.fromarray(enhanced_image.astype(np.uint8))

def enhance_contrast(image, radius):
    radius = int(radius)
    b = np.zeros(image.shape)
    cv2.circle(b, (radius, radius), int(radius * 0.9), (1, 1, 1), -1, 8, 0)
    blurred = cv2.GaussianBlur(image, (0, 0), radius / 30)
    return cv2.addWeighted(image, 4, blurred, -4, 128)*b + 128*(1 - b)

readonly_source_directory = None
readonly_target_directory = None
def set_readonly_directories(source, target):
    global readonly_source_directory
    global readonly_target_directory
    readonly_source_directory = source
    readonly_target_directory = target

# Given a filename, this will build a path for the file in our
# target directory and assign the correct extension.
def build_image_target_path(filename):
    target_filename = ''.join(filename.split('.')[:-1] + ['.jpeg'])
    return os.path.join(readonly_target_directory, target_filename)

# This will take a path-filename pair, and write
# a processed image to our target directory.
def process_image(source_image):
    source_path, source_filename = source_image[0], source_image[1]
    target_path = build_image_target_path(source_filename)
    print('Target path: ' + target_path)
    if not os.path.exists(target_path):
        img = convert(source_path, CROP_SIZE)
        if img is not None:
            img.save(target_path, quality=97)

# Entry point.
def main():
    ensure_directory_exists(readonly_target_directory)
    source_images = collect_images(readonly_source_directory)
    if not source_images:
        print('Nothing to process.')
        exit()

    print('Preprocessing initiated.')
    print('  Source: {}'.format(readonly_source_directory))
    print('  Target: {}'.format(readonly_target_directory))

    # This is actually ceil(len(filenames) / BATCH_SIZE)
    batch_count = (len(source_images) + BATCH_SIZE - 1) // BATCH_SIZE
    thread_pool = multiprocessing.pool.Pool(THREAD_COUNT, set_readonly_directories,
                                            (readonly_source_directory, readonly_target_directory))
    for i in range(batch_count):
        print('Assigning batch {}/{}'.format(i + 1, batch_count))
        thread_pool.map(process_image, source_images[i*BATCH_SIZE : (i+1)*BATCH_SIZE])
    thread_pool.close()
    print('Preprocessing complete.')

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage:')
        print('  python preprocess_images.py <source_dir> <target_dir>')
        exit()
    set_readonly_directories(sys.argv[1], sys.argv[2])
    main()
