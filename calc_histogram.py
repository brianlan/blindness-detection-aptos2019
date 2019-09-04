import argparse
from pathlib import Path
import multiprocessing

import cv2
import matplotlib

from src.helpers import calc_hist, BlackMarginCrop

matplotlib.use('tkagg')
from tqdm import tqdm

from lib.lanutils.fs.indices import get_indices

parser = argparse.ArgumentParser()
parser.add_argument("--dataset-dir", type=Path, required=True)
parser.add_argument("--indices", type=Path)
parser.add_argument("--random-sample", type=int)
parser.add_argument("--im-suffix", default=".png")
parser.add_argument("--min-value", default=0, help="min value take into account when calc hist.")
parser.add_argument("--num-processes", type=int, default=0)
args = parser.parse_args()


def get_crop_resized_img(file_index):
    im_path = args.dataset_dir / f"{file_index}{args.im_suffix}"
    im = cv2.imread(str(im_path))
    im = BlackMarginCrop(thresh=15)(im)
    im = cv2.resize(im, (224, 224))
    return im


def main():
    indices = get_indices(args.indices, args.dataset_dir, args.im_suffix, random_sample=args.random_sample)
    if args.num_processes == 0:
        ims = [get_crop_resized_img(ind) for ind in tqdm(indices)]
    else:
        with multiprocessing.Pool(processes=args.num_processes) as pool:
            ims = list(tqdm(pool.imap_unordered(get_crop_resized_img, indices), total=len(indices)))

    density = calc_hist(ims, density=True)


if __name__ == "__main__":
    main()
