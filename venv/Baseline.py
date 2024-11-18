import pandas as pd
import numpy as np
import os
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

data = pd.read_csv('venv/data/MIMIC-IV-ECG-Ext-Electrolytes/baselines/carry_forward_24h-all_obtained/mimiciv_ECGv1.1_hospV2.2_Calcium50893.csv')
results = data['flag']
predictions = data['carry_forward_lab_flag_last']
results = results.replace({'abnormal': 1, np.nan: 0})
predictions = predictions.replace({'abnormal': 1, np.nan: 0})
accuracy = accuracy_score(results, predictions)

auroc = roc_auc_score(results, predictions)

f1 = f1_score(results, predictions)

print(f"Accuracy: {accuracy}")
print(f"AUROC: {auroc}")
print(f"F1 Score: {f1}")