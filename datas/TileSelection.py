"""
1. Calculate indexes of tiles, including pooled intensity and 1D image entropy.
2. Select tiles according to indexes.

"""

from metimage.utils.ImageProcessing import SplitTiles
import pickle
from scipy import sparse
import glob
import os
import pandas as pd
import numpy as np
import tqdm
import cv2

def calEntropy(img):
    """
    Calculate 1D image entropy

    :param img: Image matrix with 2D (224x224)
    :return: E: 1D image entropy
    """
    img = np.array(img)
    img = img.astype(np.uint8)
    if img.sum()==0:
        return 0
    else:
        hist_cv = cv2.calcHist([img], [0], None, [img.max()], [0, img.max()])
        P = hist_cv / (len(img) * len(img[0]))
        P[P == 0] = 1
        E = np.sum([p * np.log2(1 / p) for p in P])
        return E

def MeanIntIndex(img):
    """
    Calculate pooled intensity

    :param img: Image matrix with 2D (224x224)
    :return: pool_int: Pooled intensity of image
    """
    pool_int = img.mean()
    return pool_int

def CalAllIndex(filelist,method="mean",pixelx=224, pixely=224, overlap_col=0, overlap_row=0,save_path="."):
    """
    Calculate pooled intensity or 1D entropy for different tiles

    :param filelist: pathway of files to calculate index
    :param method: method used (mean: pooled intensity; entropy: 1D image entropy)
    :param pixelx, pixely, overlap_col, overlap_row: refer to SplitTiles
    :param save_path: pathway of output data.
    """
    for file in tqdm.tqdm(filelist):
        filename = os.path.splitext(os.path.basename(file))[0]
        print(filename)
        SparseTable = sparse.load_npz(file)
        RawImage = pd.DataFrame(SparseTable.todense())
        tiles = SplitTiles(RawImage, pixelx=pixelx, pixely=pixely, overlap_col=overlap_col, overlap_row=overlap_row)
        tiles = np.moveaxis(tiles, 2, 0)
        entropy_list = []
        for tile in tqdm.tqdm(tiles):
            if method == "mean":
                entropy_list.append(MeanIntIndex(tile))
            elif method == "1DEntropy":
                entropy_list.append(calEntropy(tile))
        if not os.path.exists(save_path + "/" + method):
            os.makedirs(save_path + "/" + method)
        with open(save_path + "/" + method + "/"+ filename + ".etp", "wb") as f:
            pickle.dump(entropy_list, f)

# dir_mean = "Z:/memberdata/Wanghongmiao/20220905 MetImage Encoding/Trainingset final/train/mean/Total"
# dir_entropy = "Z:/memberdata/Wanghongmiao/20220905 MetImage Encoding/Trainingset final/train/1DEntropy/Total"
# TopMean = 1000
# TopEntropy = 1000

def SelectTiles(dir_mean, dir_entropy, TopMean=1000, TopEntropy=1000,save_path="."):
    """
    Select information enriched tiles by top pooled intensity and 1D image entropy

    :param dir_mean: pathway of calculated pooled intensity (.etp)
    :param dir_entropy: pathway of calculated 1D image entropy (.etp)
    :param TopMean: Top N pooled intensity selected.
    :param TopEntropy: Top N entropy selected.
    :param save_path: pathway of output data.
    :return A list of index of selected tiles
    """
    if TopMean is None and TopEntropy is None:
        print("TopMean and TopEntropy can't all be None!")
        return None
    if TopEntropy is not None:
        rawList = glob.glob(dir_entropy + "/*.etp")
        for IndexList in rawList:
            with open(IndexList, 'rb') as f:
                entropy_list = pickle.load(f)
            if IndexList == rawList[0]:
                entropy_list_all = pd.DataFrame(entropy_list, columns=[IndexList])
            else:
                entropy_list_all.insert(entropy_list_all.shape[1], IndexList, entropy_list)
        Entropy_Mean = entropy_list_all.mean(axis=1)

    if TopMean is not None:
        rawList = glob.glob(dir_mean + "/*.etp")
        for IndexList in rawList:
            with open(IndexList, 'rb') as f:
                int_list = pickle.load(f)
            if IndexList == rawList[0]:
                int_list_all = pd.DataFrame(int_list, columns=[IndexList])
            else:
                int_list_all.insert(int_list_all.shape[1], IndexList, int_list)

        Int_Mean = int_list_all.mean(axis=1)


    if TopMean is None:
        SamplingList = Entropy_Mean.sort_values(ascending=False).index[range(TopEntropy)]
    elif TopEntropy is None:
        SamplingList = Int_Mean.sort_values(ascending=False).index[range(TopMean)]
    elif TopEntropy is not None and TopMean is not None:
        index_list_etp = Entropy_Mean.sort_values(ascending=False).index[range(TopEntropy)]
        index_list_int = Int_Mean.sort_values(ascending=False).index[range(TopMean)]
        SamplingList = list(set(index_list_etp).intersection(set(index_list_int)))

    with open(save_path + "/Samplinglist.lst", "wb") as f:
        pickle.dump(SamplingList, f)

    return SamplingList

def GenerateIndex(dataset_dir,cal_mean=True,cal_entropy = True,pixelx=224, pixely=224, overlap_col=0, overlap_row=0,save_path="."):
    """
    Generate pooled intensity and 1D image entropy of dataset.

    :param dataset_dir: pathway of dataset
    :param cal_mean, cal_entropy: calculate pooled intensity or 1D image entropy (bool)
    :param pixelx, pixely, overlap_col, overlap_row: refer to SplitTiles
    :param save_path: pathway of output data.
    :return entropy or pooled intensity data.

    """
    filelist = glob.glob(dataset_dir+"/**/*.npz",recursive=True)
    if cal_mean:
        print("Calculate pooled intensity.")
        CalAllIndex(filelist,method="mean",pixelx=pixelx, pixely=pixely, overlap_col=overlap_col,
                    overlap_row=overlap_row,save_path=save_path)
    if cal_entropy:
        print("Calculate 1D image entropy.")
        CalAllIndex(filelist,method="1DEntropy",pixelx=pixelx, pixely=pixely, overlap_col=overlap_col,
                    overlap_row=overlap_row,save_path=save_path)



