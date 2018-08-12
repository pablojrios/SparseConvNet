import pandas as pd
import os
import shutil
from functools import wraps
from time import time


def timing(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time()
        result = func(*args, **kwargs)
        end = time()
        print('Elapsed time func: "{}" run in {}s'.format(func.__name__, end - start))
        return result
    return wrapper


source_data_folder = '/media/pablo/Ophthalmology/Kaggle/DR/train'
base_data_folder = '/home/pablo/dev/pablojrios-SparseConvNet/Data/kaggleDiabeticRetinopathy'
df_train_labels = pd.read_csv(base_data_folder + '/train/trainLabels.csv')

# crear los folders de las clases
os.chdir(base_data_folder + '/train')
for x in range(0, 5):
    if os.path.exists(base_data_folder + '/train/' + str(x)):
        shutil.rmtree(base_data_folder + '/train/' + str(x))
    os.mkdir(str(x))


@timing
def createSymlinkFor():
    # row[1] = '10_left', row[2] = '0'
    for row in df_train_labels.itertuples():
        os.symlink(source_data_folder + "/" + row[1] + '.jpeg',
                   base_data_folder + '/train/' + str(row[2]) + "/" + row[1] + '.jpeg')


@timing
def createSymlinkList():
    # os.symlink(src, dst)
    [os.symlink(source_data_folder + "/" + row[1] + '.jpeg',
                base_data_folder + '/train/' + str(row[2]) + "/" + row[1] + '.jpeg')
     for row in df_train_labels.itertuples()]


print(createSymlinkList())