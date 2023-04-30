import os
import time
import random
import tensorflow as tf
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import optimizers, metrics
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model
from helpers import PlotROCCurve

from dataset_path import output_path

path = output_path
output_path = os.path.join(path, "Figure3")
df_train = pd.read_csv((os.path.join(path, 'train.csv')))
df_test = pd.read_csv((os.path.join(path, 'test.csv')))

train_cxr_note_embs = list(np.load((os.path.join(path, 'train_cxr_note_embs.npy'))))
test_cxr_note_embs = list(np.load((os.path.join(path, 'test_cxr_note_embs.npy'))))

train_cxr_img_embs = list(np.load((os.path.join(path, 'train_cxr_img_embs.npy'))))


train_cxr_note_embs = pd.DataFrame(train_cxr_note_embs, columns = ['cxr_note_emb'+str(i) for i in range(128)])
test_cxr_note_embs = pd.DataFrame(test_cxr_note_embs, columns = ['cxr_note_emb'+str(i) for i in range(128)])

train_cxr_img_embs = pd.DataFrame(train_cxr_img_embs, columns = ['cxr_img_emb'+str(i) for i in range(1376)])

df_train = pd.concat([df_train, train_cxr_note_embs, train_cxr_img_embs], axis = 1)
df_test = pd.concat([df_test, test_cxr_note_embs], axis = 1)
confidence_interval = 95
random_seed=0

random.seed(random_seed)
np.random.seed(random_seed)

pd.set_option('display.max_columns', 100) 
pd.set_option('display.max_rows', 100) 
print(df_train.head())
print('training size =', len(df_train), ', testing size =', len(df_test))