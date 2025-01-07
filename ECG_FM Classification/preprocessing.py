import numpy as np
import pandas as pd
import os

import wfdb
from scipy.io import savemat
from pathos.multiprocessing import ProcessingPool as Pool
from typing import List

def create_records(raw_path, save_path, dataset):
    create_save_dir(save_path, dataset)
    save_path = save_path + dataset + '/segmented/'

    df = pd.read_csv(raw_path + 'MIMIC-IV-ECG-Ext-Electrolytes/few_shot_splits/128shots/split1/' + dataset + '.csv')
    meta_df = pd.read_csv(raw_path + 'MIMIC-IV-ECG-Ext-Electrolytes/mimiciv_ECGv1.1_hospV2.2_Calcium50893.csv')
    
    df['raw_path'] = df['study_id'].map(meta_df.set_index('study_id')['path'].apply(lambda x: raw_path + x))
    df['idx'] = np.arange(len(df))
    df['save_path'] = df['study_id'].apply(lambda x: os.path.join(save_path, f"{x}.mat"))
   
    df = df.loc[:, ~df.columns.str.contains('Unnamed: 0')]
    
    df['save_path'] = df['save_path'].apply(lambda x: x[:-4] + '_0.mat')
    df_duplicated = df.copy()
    df_duplicated['save_path'] = df_duplicated['save_path'].apply(lambda x: x[:-6] + '_1.mat')
    df = pd.concat([df, df_duplicated], ignore_index=True)
   
    df.to_csv(save_path[:-11] + '/records.csv', index=False)
   
    return df


def create_save_dir(save_path, dataset):
    preprocessed_path = save_path + dataset + '/preprocessed/'
    os.makedirs(preprocessed_path, exist_ok=True)
    segmented_path = save_path + dataset + '/segmented/'
    os.makedirs(segmented_path, exist_ok=True)
    output_path = save_path + dataset + '/output/'
    os.makedirs(output_path, exist_ok=True)
    manifest_path = save_path + dataset + '/manifest/'
    os.makedirs(manifest_path, exist_ok=True)


def create_labels(raw_path, save_path, dataset):
    create_save_dir(save_path, dataset)
    
    records = pd.read_csv(os.path.join(save_path, dataset + '/records.csv'))
    meta_df = pd.read_csv(raw_path + 'MIMIC-IV-ECG-Ext-Electrolytes/mimiciv_ECGv1.1_hospV2.2_Calcium50893.csv')
    
    labels_df = pd.DataFrame()
    labels_df['idx'] = records['idx']
    labels_df['abnormal'] = records['study_id'].map(meta_df.set_index('study_id')['flag'].apply(lambda x: x == 'abnormal'))
    
    labels_df = labels_df.drop_duplicates()

    labels_df.to_csv(save_path + dataset + '/labels.csv', index=False)

    return labels_df


def create_mat_files(save_path, dataset):
    create_save_dir(save_path, dataset)

    records = pd.read_csv(os.path.join(save_path, dataset + '/records.csv'))

    with Pool() as pool:
        results = pool.map(process_row, records.to_dict('records'))

    return results


def process_row(row):
    signal, fields = wfdb.rdsamp(row['raw_path'])
    aVL_index = fields['sig_name'].index('aVL')
    aVF_index = fields['sig_name'].index('aVF')

    signal[:, [aVL_index, aVF_index]] = signal[:, [aVF_index, aVL_index]]

    if row['save_path'].endswith('_0.mat'):
        # correct_ordering = np.array(['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'])

        # leads_to_load = pd.DataFrame(index=correct_ordering)
        # signal, _ = reorder_leads(signal, fields['sig_name'], leads_to_load)
        
        preprocessed_data = {
            "org_sample_rate": 500,
            "curr_sample_rate": 500,
            "org_sample_size": 5000,
            "curr_sample_size": 5000,
            "feats": signal,
            "idx": row['idx']
        }

        segmented_save_path = row['save_path']
        preprocessed_save_path = segmented_save_path.replace('segmented', 'preprocessed').replace('_0.mat', '.mat')
        savemat(preprocessed_save_path, preprocessed_data)
        signal = signal[:signal.shape[0] // 2]

        segmented_data = {
            "org_sample_rate": 500,
            "curr_sample_rate": 500,
            "org_sample_size": 5000,
            "curr_sample_size": 2500,
            "feats": signal,
            "idx": row['idx']
        }

        savemat(segmented_save_path, segmented_data)

    elif row['save_path'].endswith('_1.mat'):
        signal = signal[signal.shape[0] // 2:]

        # correct_ordering = np.array(['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'])

        # leads_to_load = pd.DataFrame(index=correct_ordering)
        # signal, _ = reorder_leads(signal, fields['sig_name'], leads_to_load)
        segmented_data = {
            "org_sample_rate": 500,
            "curr_sample_rate": 500,
            "org_sample_size": 5000,
            "curr_sample_size": 2500,
            "feats": signal,
            "idx": row['idx']
        }

        savemat(row['save_path'], segmented_data)
    
    
    return segmented_data
    

# def reorder_leads(feats: np.ndarray, sig_name: List[str], leads_to_load: pd.DataFrame):
#     sig_name = np.array(sig_name)

#     # If already identical, simply return feats as is
#     if np.array_equal(leads_to_load.index, sig_name):
#         return feats, leads_to_load.index.values.tolist()


#     feats_order = leads_to_load.join(
#         pd.Series(np.arange(len(sig_name)), index=sig_name, name='sample_order'),
#         how='left',
#     )

#     lead_missing = feats_order['sample_order'].isna()

#     # If no missing leads, simply re-order the leads
#     if not lead_missing.any():
#         feats = feats[feats_order['sample_order'].astype(int)]
#         return feats, leads_to_load.index.values.tolist()

#     # Otherwise, create a whole new array and fill in the available leads
#     feats_new = np.full((len(leads_to_load), feats.shape[1]), np.nan)

#     avail = feats_order[~lead_missing].astype(int)
#     for _, row in avail.iterrows():
#         feats_new[avail['global_order']] = feats[avail['sample_order']]

#     return feats_new, lead_missing.index[~lead_missing].values.tolist()

def create_meta(raw_path, save_path, dataset):
    create_save_dir(save_path, dataset)
    
    records = pd.read_csv(os.path.join(save_path, dataset + '/records.csv'))
    meta_df = pd.read_csv(raw_path + 'MIMIC-IV-ECG-Ext-Electrolytes/mimiciv_ECGv1.1_hospV2.2_Calcium50893.csv')
    
    meta_df = meta_df[meta_df['study_id'].isin(records['study_id'])]
    meta_df = meta_df.merge(records[['study_id', 'idx']], on='study_id', how='left')
    meta_df = meta_df.drop(columns=['path'])
    meta_df = meta_df.drop(columns=['itemid'])
    meta_df = meta_df.drop(columns=['valueuom'])

    meta_df = meta_df.drop_duplicates()

    meta_df.to_csv(save_path + dataset + '/meta.csv', index=False)
    return meta_df


def create_meta_split(save_path, dataset):
    create_save_dir(save_path, dataset)
    
    records = pd.read_csv(os.path.join(save_path, dataset + '/records.csv'))
    

    meta_split_df = pd.DataFrame()
    meta_split_df['idx'] = records['idx']
    meta_split_df['save_file'] = records['study_id'].apply(lambda x: f"{x}.mat")
    meta_split_df['split'] = records['split']

    meta_split_df.to_csv(save_path + dataset + '/meta_split.csv', index=False)

    return meta_split_df

def create_segmented_split(save_path, dataset):
    create_save_dir(save_path, dataset)
    
    records = pd.read_csv(os.path.join(save_path, dataset + '/records.csv'))
    
    segmented_split_df = pd.DataFrame()
    segmented_split_df['idx'] = records['idx']
    segmented_split_df['save_file'] = records['study_id'].apply(lambda x: f"{x}.mat")
    segmented_split_df['split'] = records['split']
    segmented_split_df['path'] = records['save_path']
    segmented_split_df['sample_size'] = 2500
    
    segmented_split_df = segmented_split_df.sort_values(by='idx')

    segmented_split_df.to_csv(save_path + dataset + '/segmented_split.csv', index=False)

    return segmented_split_df

def prepare_clf_labels(save_path, dataset):
    return (
        f"""python3 fairseq-signals/scripts/prepare_clf_labels.py \\
        --output_dir '{save_path}{dataset}/output/ \\
        --labels '{save_path}{dataset}/labels.csv' \\
        --meta_splits '{save_path}{dataset}/meta_split.csv' \\
        --segmented_splits '{save_path}{dataset}/segmented_split.csv'"""
    )

def prepare_manifest(save_path, dataset):
    return (
        f"""python3 fairseq-signals/scripts/manifests.py \\
        --split_file_paths '{save_path}{dataset}/segmented_split.csv' \\
        --save_dir '{save_path}{dataset}/manifest/'"""
    )

def prepare_cmsc_manifest(save_path, dataset):
    return (
        f"""python3 fairseq-signals/fairseq_signals/data/ecg/preprocess/convert_to_cmsc_manifest.py \\
        '{save_path}{dataset}/manifest/' \\
        --dest '{save_path}{dataset}/manifest/'"""
    )

def make_inference(raw_path, save_path, dataset):
    return (
        f"""fairseq-hydra-inference \\
        task.data="{save_path}{dataset}/manifest/cmsc" \\
        common_eval.path="{raw_path}ckpts/mimic_iv_ecg_physionet_pretrained.pt" \\
        common_eval.results_path="{save_path}{dataset}/outputs" \\
        dataset.valid_subset="test" \\
        dataset.batch_size=10 \\
        dataset.num_workers=3 \\
        dataset.disable_validation=false \\
        distributed_training.distributed_world_size=1 \\
        distributed_training.find_unused_parameters=True \\
        --config-dir "{raw_path}ckpts" \\
        --config-name mimic_iv_ecg_physionet_pretrained
        """
    )