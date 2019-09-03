import argparse
from pathlib import Path
import multiprocessing

import cv2
import numpy as np
from tqdm import tqdm
import pandas as pd

from lib.lanutils.helpers import get_indices


parser = argparse.ArgumentParser()
parser.add_argument("--dataset-dir", type=Path, required=True)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--label", type=Path)
parser.add_argument("--indices", type=Path)
parser.add_argument("--im-suffix", default=".png")
parser.add_argument("--num-processes", type=int, default=1)
args = parser.parse_args()
label = pd.read_csv(args.label, index_col="image").to_dict()["label"]
id2cls = {0: "no_dr", 1: "mild", 2: "moderate", 3: "severe", 4: "proliferative_dr"}


def draw_class_text(
    im, cls, position, font=cv2.FONT_HERSHEY_DUPLEX, font_scale=1.0, color=(255, 255, 255), thickness=1
):
    return cv2.putText(im, cls, position, font, font_scale, color, thickness)


def vis_image(file_index):
    rel_im_path = f"{file_index}{args.im_suffix}"
    im_path = args.dataset_dir / rel_im_path
    cls = id2cls[label[im_path.stem]]
    save_path = args.output_dir / cls / rel_im_path
    save_path.parent.mkdir(parents=True, exist_ok=True)
    im = cv2.imread(str(im_path))
    im = cv2.resize(im, (224, 224))
    im = draw_class_text(im, cls, (20, 20))
    cv2.imwrite(str(save_path), im)


def read_im(file_index):
    im_path = args.dataset_dir / f"{file_index}{args.im_suffix}"
    im = cv2.imread(str(im_path))
    im = cv2.resize(im, (224, 224))
    return im


def get_dataset_mean_std(ims):
    ims = np.array(ims)
    mu = ims.sum(axis=0).sum(axis=0).sum(axis=0) / (len(ims) * 224 * 224)
    var = ((ims - mu[None, None, None, :]) ** 2).sum(axis=0).sum(axis=0).sum(axis=0) / (len(ims) * 224 * 224)
    std = np.sqrt(var)
    print(mu / 255.0)
    print(std / 255.0)


def main():
    indices = get_indices(args.dataset_dir, args.im_suffix, indices_path=args.indices)
    # for ind in tqdm(indices):
    #     vis_image(ind)
    with multiprocessing.Pool(processes=args.num_processes) as pool:
        # _ = list(tqdm(pool.imap_unordered(vis_image, indices), total=len(indices)))
        ims = list(tqdm(pool.imap_unordered(read_im, indices), total=len(indices)))

    get_dataset_mean_std(ims)


if __name__ == "__main__":
    main()
