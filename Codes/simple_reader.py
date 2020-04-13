import pandas as pd 
import numpy as np 
import os

f = 'File_references.csv'

file_df = pd.read_csv(f)
file_df.head()

subject_ID = file_df['Sample'].to_numpy()
file_name = file_df['File name'].to_numpy()
y = file_df['Perforation'].to_numpy()


    
def simple_reader(file_name, subject_ID, y):
    for f, ID, y in zip(file_name,subject_ID,y):
        img = sitk.GetArrayFromImage(sitk.ReadImage(f))
        img = img / 255.0
        img = np.expand_dims(img, axis = -1).astype(np.float32)

        y = np.expand_dims(y, axis = -1).astype(np.float32)

        yield {'features':img, 'lable':y, 'Sample_ID':ID}