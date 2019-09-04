import argparse
from pathlib import Path
import multiprocessing

import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import cv2
import numpy as np
from tqdm import tqdm

from lib.lanutils.fs.indices import get_indices

parser = argparse.ArgumentParser()
parser.add_argument("--dataset-dir", type=Path, required=True)
parser.add_argument("--indices", type=Path)
parser.add_argument("--random-sample", type=int)
parser.add_argument("--im-suffix", default=".png")
parser.add_argument("--num-processes", type=int, default=0)
args = parser.parse_args()


class BlackMarginCrop(object):
    def __init__(self, *, thresh=1, **kwargs):
        """
        :param thresh: used to determine the binary image, i.e. pixel value lt thresh goes to 0
        """
        self.thresh = thresh

    def __call__(self, im):
        left = top = 0
        h, w = im.shape[:2]
        right, bottom = w - 1, h - 1
        im_grayscale = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        if im_grayscale.sum() == 0:
            raise ValueError("Encountered pure black image. ")
        im_grayscale[im_grayscale < self.thresh] = 0
        squashed_row = im_grayscale.sum(axis=0)
        squashed_col = im_grayscale.sum(axis=1)
        row_nonzero = (squashed_row / h > 0.1).nonzero()[0]
        col_nonzero = (squashed_col / w > 0.1).nonzero()[0]
        if len(row_nonzero) > 0:
            left, right = row_nonzero[0], row_nonzero[-1]
        if len(col_nonzero) > 0:
            top, bottom = col_nonzero[0], col_nonzero[-1]
        return im[top:bottom+1, left:right+1].copy()


def get_crop_resized_imgs(file_index):
    im_path = args.dataset_dir / f"{file_index}{args.im_suffix}"
    im = cv2.imread(str(im_path))
    im = BlackMarginCrop(thresh=15)(im)
    im = cv2.resize(im, (224, 224))
    return im


def get_dataset_mean_std(ims):
    ims = np.array(ims)
    mu = ims.sum(axis=0).sum(axis=0).sum(axis=0) / (len(ims) * 224 * 224)
    var = ((ims - mu[None, None, None, :]) ** 2).sum(axis=0).sum(axis=0).sum(axis=0) / (len(ims) * 224 * 224)
    std = np.sqrt(var)
    print(mu / 255.0)
    print(std / 255.0)


def calc_hist(ims):
    pix_by_channel = np.array(ims).transpose(3, 1, 2, 0).reshape(3, -1)
    hist = [np.histogram(c, bins=255)[0] for c in pix_by_channel]
    return hist


def main():
    indices = get_indices(args.indices, args.dataset_dir, args.im_suffix, random_sample=args.random_sample)
    if args.num_processes == 0:
        ims = [get_crop_resized_imgs(ind) for ind in tqdm(indices)]
    else:
        with multiprocessing.Pool(processes=args.num_processes) as pool:
            ims = list(tqdm(pool.imap_unordered(get_crop_resized_imgs, indices), total=len(indices)))

    hist = calc_hist(ims)


if __name__ == "__main__":
    main()
