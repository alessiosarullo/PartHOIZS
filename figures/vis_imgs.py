from lib.dataset.hicodet_hake import HicoDetHake
from lib.dataset.vcoco import VCoco
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


def main():
    split = 'train'
    # ds = HicoDetHake()  # 1080, 5968, 6291, 8199, 13928, 18053?, 19014, 19708, 20200; 6190
    ds = VCoco()  # 2735; 114 (Gkioxari)
    gt = ds.get_img_data(split)

    for i, x in enumerate(gt):
        if x.labels is None:
            continue
        # istr = [ds.interactions_str[j] for j in x.labels]
        # if any(['cut ' in y or 'cut_with knife' in y for y in istr]):

        # box_classes = [ds.objects[x.box_classes[int(y)]] if not np.isnan(y) else 'None' for y in x.ho_pairs[:, 1]]
        # istr = [f'{ds.actions[j]} {o}' for j, o in zip(x.labels, box_classes)]
        # if any(['cut_obj' in y or 'cut_instr knife' in y for y in istr]):

        # if i in [8199]:
        if i in [2735]:
            print(i, x.filename)
            plt.figure(figsize=(16, 10))
            img_path = ds.get_img_path(split, x.filename)
            img = Image.open(img_path)
            plt.imshow(img)
            plt.show()


if __name__ == '__main__':
    main()
