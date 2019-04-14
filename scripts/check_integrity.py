'''
Ensures that our long-running processes for preprocessing
images actually did what they were supposed to do.
'''

import sys
from scripts.fs_utils import collect_images

def main(source_dir, target_dir):
    source_filenames = [filename for (filepath, filename) in collect_images(source_dir)]
    target_filenames = [filename for (filepath, filename) in collect_images(target_dir)]
    source_filenames.sort()
    target_filenames.sort()
    ok = True
    print('Source count: ' + str(len(source_filenames)))
    print('Target count: ' + str(len(target_filenames)))
    if len(source_filenames) != len(target_filenames):
        print('Sizes mismatch...')
        ok = False
    for sfname, tfname in zip(source_filenames, target_filenames):
        if sfname != tfname:
            print('Mismatched!')
            print(' Source: ' + sfname)
            print(' Target: ' + tfname)
            ok = False
            break
    if ok:
        print('Seems everything is okay.')
    else:
        print('Failed, things are not okay.')

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('This script is meant to ensure that preprocess_images.py got all')
        print('the files written to the target directory.')
        print('')
        print('Usage:')
        print('   python check_integrity.py <source_dir> <target_dir>')
    source_dir = sys.argv[1]
    target_dir = sys.argv[2]
    main(source_dir, target_dir)
