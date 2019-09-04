import cv2
import numpy as np


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
        return im[top : bottom + 1, left : right + 1].copy()


def get_dataset_mean_std(ims):
    ims = np.array(ims)
    mu = ims.sum(axis=0).sum(axis=0).sum(axis=0) / (len(ims) * 224 * 224)
    var = ((ims - mu[None, None, None, :]) ** 2).sum(axis=0).sum(axis=0).sum(axis=0) / (len(ims) * 224 * 224)
    std = np.sqrt(var)
    print(mu / 255.0)
    print(std / 255.0)


def calc_hist(ims, density=False, min_value=6):
    pix_by_channel = np.array(ims).transpose(3, 1, 2, 0).reshape(3, -1)
    hist = [np.histogram(c, bins=255-min_value, density=density, range=(min_value, 255))[0] for c in pix_by_channel]
    return hist


def get_density_mean(density_channel):
    """density_channel sum to 1, so integral the density and see which position right exceeds 0.5"""
    return (np.cumsum(density_channel) - 0.5 > 0).nonzero()[0][0]


def similar_to_2019(density):
    red_mean = get_density_mean(density[2])
    green_mean = get_density_mean(density[1])
    blue_mean = get_density_mean(density[0])
    return (
        110 < red_mean < 210
        and 40 < green_mean < 100
        and blue_mean < 60
        and 20 < red_mean - green_mean < 80
        and 10 < green_mean - blue_mean < 50
    )
