
import os

def ensure_directory_exists(directory):
    try:
        os.mkdir(directory)
    except OSError:
        pass

def is_image_filename(filename):
    return filename.lower().split('.')[-1] in ['jpg', 'jpeg', 'tif', 'png']

# This function will construct a list of image
# (filepath, filename) pairs from a given directory.
def collect_images(directory):
    source_images = []
    for directory_prefix, _, filenames in os.walk(directory):
        image_filenames = [filename for filename in filenames if is_image_filename(filename)]
        source_images.extend([(os.path.join(directory_prefix, f), f) for f in image_filenames])
    return source_images
