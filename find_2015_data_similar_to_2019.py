import shutil
import argparse
from pathlib import Path
import multiprocessing

import cv2
from tqdm import tqdm

from lib.lanutils.fs.indices import get_indices
from src.helpers import similar_to_2019, BlackMarginCrop, calc_hist

parser = argparse.ArgumentParser()
parser.add_argument("--dataset-dir", type=Path, required=True)
parser.add_argument("--result-save-dir", type=Path, required=True)
parser.add_argument("--copy-images", default=False, action="store_true")
parser.add_argument("--indices", type=Path)
parser.add_argument("--random-sample", type=int)
parser.add_argument("--im-suffix", default=".jpeg")
parser.add_argument("--min-value", default=6, type=int, help="min value take into account when calc hist.")
parser.add_argument("--seed", type=int)
parser.add_argument("--num-processes", type=int, default=0)
args = parser.parse_args()


def find_similar_img(file_index):
    im_path = args.dataset_dir / f"{file_index}{args.im_suffix}"
    im = cv2.imread(str(im_path))
    im = BlackMarginCrop(thresh=15)(im)
    im = cv2.resize(im, (224, 224))
    density = calc_hist(im[None], density=True, min_value=args.min_value)
    if similar_to_2019(density):
        if args.copy_images:
            dest_path = args.result_save_dir / f"{file_index}{args.im_suffix}"
            shutil.copy(im_path, dest_path)
        return file_index
    return


def main():
    args.result_save_dir.mkdir(parents=True, exist_ok=True)
    indices = get_indices(
        args.indices, args.dataset_dir, args.im_suffix, random_sample=args.random_sample, seed=args.seed
    )
    if args.num_processes == 0:
        similar_file_indices = [find_similar_img(ind) for ind in tqdm(indices)]
    else:
        with multiprocessing.Pool(processes=args.num_processes) as pool:
            similar_file_indices = list(tqdm(pool.imap_unordered(find_similar_img, indices), total=len(indices)))

    similar_file_indices = [i for i in similar_file_indices if i is not None]
    print(f"{len(similar_file_indices)} similar images was found.")
    with open(args.result_save_dir / "similar_images.txt", "w") as f:
        f.write("\n".join(similar_file_indices))


if __name__ == "__main__":
    main()
