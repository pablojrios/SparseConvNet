import pandas as pd
base_data_folder = '/home/pablo/dev/SparseConvNet/Data/kaggleDiabeticRetinopathy'
df1 = pd.read_csv(base_data_folder + '/train/trainLabels.csv')


def left(s, amount):
    """left substring slice function"""
    return s[:amount]


def right(s, amount):
    """right substring slice function"""
    return s[-amount:]


def mid(s, offset, amount):
    """middle substring slice function"""
    return s[offset:offset+amount]


def patient_sample(df_in, number_patients):
    # Notes to remember:
    # df[['colname']] returns a new DataFrame, while df['colname'] returns a Series
    # map() works on Series objects, apply() works on DataFrame objects

    # copia de los labels de training
    df_left = df_in.copy()
    # agrego columna 'is_left_eye' indicando si es ojo izquierdo (True) o derecho (False)
    df_left['is_left_eye'] = df_in['image'].map(lambda image: right(image, len('left')) == 'left')
    # otra forma de hacer lo anterior
    # df_left['left_eye'] = df_in[['image']].apply(lambda image: right(image.str, len('left')) == 'left')
    # nuevo df con las imágenes ojo izquierdo
    df_left = df_left[df_left['is_left_eye']]
    # remuevo columna flag 'is_left_eye'
    df_left = df_left.drop('is_left_eye', axis=1)
    seed = 12345
    # Randomly sample 70% of your dataframe
    # df_70p = df_in.sample(frac=0.7)
    # Randomly sample 100 elements from your dataframe
    df_left_sample = df_left.sample(n=number_patients, random_state=seed)
    # armo una lista (serie) con los nombres de las imagenes de ojos derecho correspondientes a la muestra
    # ojos izquierdo; necesito hacer esto ya que los levels ojos misma persona puede diferir
    serie_right_sample = df_left_sample['image'].map(lambda image: image[:-len('_left')]+'_right')
    # tomo del data frame source la muestra ojos derecho
    df_right_sample = df_in[df_in['image'].isin(serie_right_sample)]
    # concateno imagenes ojo izquierdo y derecho
    df_sample = pd.concat([df_left_sample, df_right_sample]).sort_values(by=['image'])
    # genero .csv muestra
    return df_sample


# Randomly sample 100 patients (ie.: 200 eyes) for training
df_train_sample = patient_sample(df1, 1000)
df_train_sample.to_csv(base_data_folder + '/train_set_1000', header=False, index=False, sep=' ')

# You can get the rest of the rows by doing:
df_rest = df1.loc[~df1.index.isin(df_train_sample.index)]
# Randomly sample 20 patients (ie.: 40 eyes) for validation
df_val_sample = patient_sample(df_rest, 200)
df_val_sample.to_csv(base_data_folder + '/val_set_200', header=False, index=False, sep=' ')

df2 = pd.read_csv(base_data_folder + '/test/retinopathy_solution.csv')
# Randomly sample 50 patients (ie.: 100 eyes) for testing
df_test_sample = patient_sample(df2, 500)
df_test_sample = df_test_sample[['image']]
df_test_sample.to_csv(base_data_folder + '/test_set_500', header=False, index=False, sep=' ')

