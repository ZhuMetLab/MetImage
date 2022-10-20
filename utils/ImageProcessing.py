"""
Tools to process converted LC-MS images.
"""

import numpy as np

def SplitTiles(MetImage, pixelx=224, pixely=224, overlap_col=0, overlap_row=0):
    """
    Split the whole MetImage into small tiles

    :param MetImage: MetImage data for splitting
    :param pixelx: the width of every tile in pixel (custom)
    :param pixely: the length of every tile in pixel (custom)
    :param overlap_col: the overlap width value of tiles split in pixel (custom)
    :param overlap_row: the overlap length value of tiles split in pixel (custom)
    :return: split tiles
    """

    window_col = pixelx - 2 * overlap_col
    window_row = pixelx - 2 * overlap_row
    window_num_col = MetImage.shape[0] // window_col + 1
    window_num_row = MetImage.shape[1] // window_row + 1
    img = MetImage
    sh = list(img.shape)
    sh[0], sh[1] = window_num_col * pixelx, window_num_row * pixely
    img_ = np.zeros(shape=sh)
    marginx = int(window_num_col * pixelx - MetImage.shape[0])
    marginy = int(window_num_row * pixely - MetImage.shape[1])
    img_[0:-marginx, 0:-marginy] = img
    tiles = []
    for i in range(window_num_col):
        start_col = overlap_col + i * window_col
        for j in range(window_num_row):
            start_row = overlap_row + j * window_row
            cropped = img_[(start_col - overlap_col):(start_col + overlap_col + window_col),
                      (start_row - overlap_row):(start_row + overlap_row + window_row)]
            tiles.append(cropped)
    tiles = np.array(tiles)
    tiles = np.moveaxis(tiles, 0, 2)
    return tiles
