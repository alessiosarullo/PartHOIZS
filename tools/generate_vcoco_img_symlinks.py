from lib.dataset.vcoco import VCoco
import os
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dst_folder', type=str)
    parser.add_argument('img_folder', type=str)
    args = parser.parse_args()
    dst_folder = args.dst_folder
    img_folder = args.img_folder

    os.makedirs(dst_folder)

    vcoco = VCoco()
    for fnames in vcoco.split_filenames.values():
        for i, fn in enumerate(fnames):
            if i % 100 == 0:
                print(f'{i + 1}/{len(fnames)}.')

            split = fn.split('_')[-2]
            split_dir = os.path.join(dst_folder, split)
            os.makedirs(split_dir, exist_ok=True)

            img_path = os.path.join(img_folder, fn)
            if not os.path.isfile(img_path):
                raise ValueError(f'File {img_path} does not exist.')
            os.symlink(img_path, os.path.join(split_dir, fn))


if __name__ == '__main__':
    main()
