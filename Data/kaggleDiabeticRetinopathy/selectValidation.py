import pandas as pd
base_data_folder='/home/pablo/dev/btgraham-SparseConvNet/Data/kaggleDiabeticRetinopathy'
df1 = pd.read_csv(base_data_folder + '/train/trainLabels.csv')
df2 = pd.read_csv(base_data_folder + '/train_minus_val_set', delim_whitespace=True, header=None)
# agrego columna boolean 'train' para indicar que imagenes están en training
df1['train'] = df1['image'].isin(df2.iloc[:,0])
# en val_set me quedo con las imagenes en validation
val_set = df1[df1['train']==False]
val_set = val_set.drop('train', axis=1)
val_set.to_csv(base_data_folder + '/val_set', header=False, index=False, sep=' ')
