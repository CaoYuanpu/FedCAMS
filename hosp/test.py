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
# df_train = pd.read_csv((os.path.join(path, 'train.csv')))
df_test = pd.read_csv((os.path.join(path, 'test.csv')))

# train_cxr_note_embs = list(np.load((os.path.join(path, 'train_cxr_note_embs.npy'))))
test_cxr_note_embs = list(np.load((os.path.join(path, 'test_cxr_note_embs.npy'))))

# train_cxr_img_embs = list(np.load((os.path.join(path, 'train_cxr_img_embs.npy'))))
test_cxr_img_embs = list(np.load((os.path.join(path, 'test_cxr_img_embs.npy'))))

# train_cxr_note_embs = pd.DataFrame(train_cxr_note_embs, columns = ['cxr_note_emb'+str(i) for i in range(128)])
test_cxr_note_embs = pd.DataFrame(test_cxr_note_embs, columns = ['cxr_note_emb'+str(i) for i in range(128)])

# train_cxr_img_embs = pd.DataFrame(train_cxr_img_embs, columns = ['cxr_img_emb'+str(i) for i in range(1376)])
test_cxr_img_embs = pd.DataFrame(test_cxr_img_embs, columns = ['cxr_img_emb'+str(i) for i in range(1376)])

# df_train = pd.concat([df_train, train_cxr_note_embs, train_cxr_img_embs], axis = 1)
df_test = pd.concat([df_test, test_cxr_note_embs, test_cxr_img_embs], axis = 1)

confidence_interval = 95
random_seed=0

random.seed(random_seed)
np.random.seed(random_seed)

pd.set_option('display.max_columns', 100) 
pd.set_option('display.max_rows', 100) 
# print(df_train.head())
# print('training size =', len(df_train), ', testing size =', len(df_test))


variable = ["age", "gender", 
            
            "n_ed_30d", "n_ed_90d", "n_ed_365d", "n_hosp_30d", "n_hosp_90d", 
            "n_hosp_365d", "n_icu_30d", "n_icu_90d", "n_icu_365d", 
            
            "triage_temperature", "triage_heartrate", "triage_resprate", 
            "triage_o2sat", "triage_sbp", "triage_dbp", "triage_pain", "triage_acuity",
            
            "chiefcom_chest_pain", "chiefcom_abdominal_pain", "chiefcom_headache",
            "chiefcom_shortness_of_breath", "chiefcom_back_pain", "chiefcom_cough", 
            "chiefcom_nausea_vomiting", "chiefcom_fever_chills", "chiefcom_syncope", 
            "chiefcom_dizziness", 
            
            "cci_MI", "cci_CHF", "cci_PVD", "cci_Stroke", "cci_Dementia", 
            "cci_Pulmonary", "cci_Rheumatic", "cci_PUD", "cci_Liver1", "cci_DM1", 
            "cci_DM2", "cci_Paralysis", "cci_Renal", "cci_Cancer1", "cci_Liver2", 
            "cci_Cancer2", "cci_HIV", 
            
            "eci_Arrhythmia", "eci_Valvular", "eci_PHTN",  "eci_HTN1", "eci_HTN2", 
            "eci_NeuroOther", "eci_Hypothyroid", "eci_Lymphoma", "eci_Coagulopathy", 
            "eci_Obesity", "eci_WeightLoss", "eci_FluidsLytes", "eci_BloodLoss",
            "eci_Anemia", "eci_Alcohol", "eci_Drugs","eci_Psychoses", "eci_Depression"]
variable.extend(list(test_cxr_note_embs.columns))
variable.extend(list(test_cxr_img_embs.columns))
print(variable)
outcome = "outcome_hospitalization"

# X_train = df_train[variable].copy()
# y_train = df_train[outcome].copy()
X_test = df_test[variable].copy()
y_test = df_test[outcome].copy()

# X_train.dtypes.to_frame().T

encoder = LabelEncoder()
# X_train['gender'] = encoder.fit_transform(X_train['gender'])
X_test['gender'] = encoder.transform(X_test['gender'])

print('class ratio')
# ratio = y_train.sum()/(~y_train).sum()
print('positive : negative =', ratio, ': 1')


# Containers for all results
result_list = []


class MLP(tf.keras.Model):
    def __init__(self):
        super(MLP, self).__init__()
        self.dense_1 = Dense(200, activation='relu')
        self.dense_2 = Dense(20, activation='relu')
        self.classifier = Dense(1, activation='sigmoid')

    def call(self, x):
        x = self.dense_1(x)
        x = self.dense_2(x)
        return self.classifier(x)


# skip this cell if not retraining
# mlp = MLP()
# mlp.compile(loss='binary_crossentropy', 
#               optimizer=optimizers.Adam(learning_rate=0.001), 
#               metrics=['accuracy', 'AUC', {'auprc': metrics.AUC(name='auprc', curve='PR')}, 
#                        'TruePositives', 'TrueNegatives', 'Precision', 'Recall'])
# start = time.time()
# mlp.fit(X_train.astype(np.float32), y_train, batch_size=200, epochs=200)
# runtime = time.time() - start
# print('Training time:', runtime, 'seconds')
# mlp.save('hospitalization_triage_mlp')

print("MLP:")
mlp = load_model('hospitalization_triage_mlp')
probs = mlp.predict(X_test.astype(np.float32))
result = PlotROCCurve(probs,y_test, ci=confidence_interval, random_seed=random_seed)