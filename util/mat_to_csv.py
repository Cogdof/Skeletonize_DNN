import scipy.io
import pandas as pd


print("mat file to csv")
mat = scipy.io.loadmat('/home/mll/v_mll3/OCR_data/dataset/IIIT5K/testdata.mat')
mat = {k:v for k, v in mat.items() if k[0] != '_'}
data = pd.DataFrame({k: pd.Series(v[0]) for k, v in mat.items()})
data.to_csv("/home/mll/v_mll3/OCR_data/dataset/IIIT5K/testdata.csv")
print("done!")