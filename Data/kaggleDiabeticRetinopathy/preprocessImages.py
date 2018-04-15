# Preprocess training images.
# Scale 300 seems to be sufficient; 500 and 1000 may be overkill
import cv2
import glob
import numpy
import pandas as pd
import os

images_source = '/media/pablo/Ophthalmology/Kaggle/DR'
if not os.path.isdir(images_source):
    print(images_source + ' directory not found!')
    exit(-1)

df1 = pd.read_csv('train_set_100', header=None, sep=' ')
# me quedo sólo con la primer columna con las imagenes
df1 = df1.iloc[:,0]
# concateno folder y extension '.jpeg'
list1 = ['train/' + image + '.jpeg' for image in df1.tolist()]
df2 = pd.read_csv('val_set_20', header=None, sep=' ')
df2 = df2.iloc[:,0]
list2 = ['train/' + image + '.jpeg' for image in df2.tolist()]
df3 = pd.read_csv('test_set_50', header=None, sep=' ')
df3 = df3.iloc[:,0]
list3 = ['test/' + image + '.jpeg' for image in df3.tolist()]


def scaleRadius(img, scale):
    x = img[int(img.shape[0]/2),:,:].sum(1)
    r = (x>x.mean()/10).sum()/2
    s = scale*1.0/r
    return cv2.resize(img,(0,0),fx=s,fy=s)


# scale in [300, 500, 1000]
for scale in [500]:
    # for f in (glob.glob("train/*.jpeg")+glob.glob("test/*.jpeg")):
    for f in (list1 + list2 + list3):
        try:
            print("preprocessing " + str(scale)+"_" + f)
            a = cv2.imread(images_source + '/' + f)
            a = scaleRadius(a, scale)
            b = numpy.zeros(a.shape)
            cv2.circle(b, (int(a.shape[1]/2), int(a.shape[0]/2)), int(scale*0.9), (1, 1, 1), -1, 8, 0)
            aa = cv2.addWeighted(a, 4, cv2.GaussianBlur(a, (0, 0), scale/30), -4, 128)*b+128*(1-b)
            # cv2.imwrite(str(scale)+"_"+f, aa, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            cv2.imwrite(str(scale)+"_"+f, aa)
        except:
            print(str(scale)+"_" + f + " failed")
