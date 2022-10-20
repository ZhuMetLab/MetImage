"""
Demo ESCC diagnosis model

"""
import metimage
from metimage.datas.DataConverter import ConvertDataset
from metimage.models.train import LoadModel
from metimage.models.train import Model_prediction
from metimage.models.train import ParseIndex

def Demo_ESCC_Diagnosis(dataset_wd=metimage.__path__ [0]+"/demo/Rawdata",
                        Sampling_lst_dir=metimage.__path__ [0]+"/checkpoints/Samplinglist.lst",
                        weight_dir= metimage.__path__ [0]+"/checkpoints/Best_weights.hdf5",
                        output=metimage.__path__ [0]+"/demo"):

    ConvertDataset(rawdata_dir=dataset_wd, pattern="mzXML", mzmin=60, mzmax=1200,
                   binSize=0.01, Threads=6, save_path=output+"/Convert")

    Sampling_list = ParseIndex(lst_dir=Sampling_lst_dir)
    model = LoadModel(Tiles_no=len(Sampling_list), weight_dir=weight_dir)
    Model_prediction(pred_dir=output+"/Convert", model=model, Sampling_list=Sampling_list,
                     print=True, save_dir=output, workers=12)

if __name__ == '__main__':
    Demo_ESCC_Diagnosis(dataset_wd=metimage.__path__[0] + "/demo/Rawdata",
                        Sampling_lst_dir=metimage.__path__[0] + "/checkpoints/Samplinglist.lst",
                        weight_dir=metimage.__path__[0] + "/checkpoints/Best_weights.hdf5",
                        output=metimage.__path__[0] + "/demo")