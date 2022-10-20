"""
A stratified sampling to split training set and validation set

"""

import os
import shutil
import glob
import pandas as pd
import random

def SplitDataset(dataset_wd, dataset_table_dir=None, test_split=0.3, seed=42):
    """
    Split dataset

    """
    Groups = os.listdir(dataset_wd)
    for Group in Groups:
        rawlist = glob.glob(dataset_wd + "/" + Group + "/*.npz")
        if dataset_table_dir is not None:
            sample_info = pd.read_csv(dataset_table_dir)
            sample_dict = dict(zip(sample_info["sample.name"], sample_info["dataset"]))
            for file in rawlist:
                filename = os.path.splitext(os.path.basename(file))[0]
                GroupLabel = sample_dict.get(filename)
                if GroupLabel is not None:
                    if not os.path.exists(
                            dataset_wd + "/" + GroupLabel + "/" + Group):
                        os.makedirs(dataset_wd + "/" + GroupLabel + "/" + Group)
                    shutil.move(file,
                                dataset_wd + "/" + GroupLabel + "/" + Group + "/" + os.path.basename(
                                    file))
        else:
            random.seed(seed)
            test_sample = random.sample(rawlist, int(test_split * len(rawlist)))
            for file in test_sample:
                GroupLabel = "validation"
                if GroupLabel is not None:
                    if not os.path.exists(
                            dataset_wd + "/" + GroupLabel + "/" + Group):
                        os.makedirs(dataset_wd + "/" + GroupLabel + "/" + Group)
                    shutil.move(file,
                                dataset_wd + "/" + GroupLabel + "/" + Group + "/" + os.path.basename(
                                    file))
            for file in list(set(rawlist) ^ set(test_sample)):
                GroupLabel = "train"
                if GroupLabel is not None:
                    if not os.path.exists(
                            dataset_wd + "/" + GroupLabel + "/" + Group):
                        os.makedirs(dataset_wd + "/" + GroupLabel + "/" + Group)
                    shutil.move(file,
                                dataset_wd + "/" + GroupLabel + "/" + Group + "/" + os.path.basename(
                                    file))
