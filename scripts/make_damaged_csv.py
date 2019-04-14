import os
from scripts.fs_utils import collect_images

def main():
    images = collect_images('../../damaged')
    with open('../../damaged/damaged.csv', 'w') as f:
        f.write('image,level\n')
        for fpath, fname in images:
            name = fname.split('.')[0]
            f.write('{},{}\n'.format(name, 0))

if __name__ == '__main__':
    main()
