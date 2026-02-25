import pandas as pd
import numpy as np
import ast
import os
import wfdb          
import neurokit2 as nk
from tqdm import tqdm

# PTB-XL dataset folder containing records100/ and the CSV
csv_records = "/Users/sheikhomeister/Library/CloudStorage/OneDrive-OsloMet/Bachelor prosjekt/Coding/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
csv_records_path = os.path.join(csv_records, "ptbxl_database.csv")
df = pd.read_csv(csv_records_path, index_col='ecg_id')


df['scp_codes_dict'] = df['scp_codes'].apply(lambda x: ast.literal_eval(x))

MI_group = ['IMI', 'ASMI', 'AMI', 'ALMI', 'ILMI', 'LMI', 'PMI', 'IPMI', 'IPLMI']

def get_MI(scp_dict):

    mi_detected = any(code in scp_dict for code in MI_group)
    normal_detected = 'NORM' in scp_dict
    if mi_detected:
        return 1
    elif normal_detected:
        return 0
    else:
        return -1

df['mi_label'] = df['scp_codes_dict'].apply(get_MI)

df_filtered = df[df['mi_label'] >= 0].copy()

df_filtered['filename'] = df_filtered['filename_lr'].apply(
    lambda x: os.path.join(csv_records, x)
)

print(f"Dataset:{len(df_filtered)} records")
print(f"MI:{(df_filtered['mi_label'] == 1).sum()}")
print(f"Normal:{(df_filtered['mi_label'] == 0).sum()}")


#Split data as mentioned in PTB-XL 
def split_data(data):
    train_mask = data['strat_fold'].isin([1, 2, 3, 4, 5, 6, 7, 8])
    val_mask   = data['strat_fold'] == 9
    test_mask  = data['strat_fold'] == 10
    return train_mask, val_mask, test_mask


#Extract the features
ALL_LEADS = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']


def avg_amplitude(indices, signal):

    valid = [int(x) for x in indices if not np.isnan(x)]
    valid = [x for x in valid if x < len(signal)]
    if not valid:
        return np.nan
    return np.nanmean(signal[valid])


def avg_duration(onsets, offsets, sampling_rate):
    
    #Formula: duration_ms = (offset_sample - onset_sample) / sampling_rate * 1000
    durations = []
    for on, off in zip(onsets, offsets):
        if not np.isnan(on) and not np.isnan(off):
            durations.append((off - on) / sampling_rate * 1000)
    return np.nanmean(durations) if durations else np.nan


def extract_MI_Features(filepath, sampling_rate=100):
    
    try:
        data, meta = wfdb.rdsamp(filepath)

        features = {}
        rpeaks_lead_ii = None

        for i, lead_name in enumerate(ALL_LEADS):
            signal = data[:, i]

            ecg_clean = nk.ecg_clean(signal, sampling_rate=sampling_rate, method="neurokit")

            _, rpeaks = nk.ecg_peaks(ecg_clean, sampling_rate=sampling_rate)

            if lead_name == 'II':
                rpeaks_lead_ii = rpeaks

            try:
                _, waves = nk.ecg_delineate(
                    ecg_clean, rpeaks,
                    sampling_rate=sampling_rate,
                    method="dwt"
                )
            except:
                for feat in ['ST_Amplitude', 'ST_Duration', 'T_amplitude', 'T_Inversion', 'Q_Amplitude', 'R_Amplitude', 'QRS_Duration', 'P_Duration']:
                    features[f'{lead_name}_{feat}'] = np.nan
                continue

            baseline = avg_amplitude(waves.get("ECG_P_Onsets", []), ecg_clean)
            if np.isnan(baseline):
                baseline = avg_amplitude(waves.get("ECG_Q_Peaks", []), ecg_clean)
            if np.isnan(baseline):
                baseline = 0

            #ST - segment Amplitude
            j_point_indices = waves.get("ECG_R_Offsets", [])
            j_point = avg_amplitude(j_point_indices, ecg_clean)
            features[f'{lead_name}_ST_Amplitude'] = (j_point - baseline) if not np.isnan(j_point) else np.nan

            #ST - segment Duration (ms)
            features[f'{lead_name}_ST_Duration'] = avg_duration(
                waves.get("ECG_R_Offsets", []),
                waves.get("ECG_T_Onsets", []),
                sampling_rate
            )

            #T - wave Amplitude
            t_peak = avg_amplitude(waves.get("ECG_T_Peaks", []), ecg_clean)
            features[f'{lead_name}_T_Amplitude'] = (t_peak - baseline) if not np.isnan(t_peak) else np.nan

            #T_Inversion
            # T-wave inversion (negative T_Amp) is a key MI indicator
            t_amp = features[f'{lead_name}_T_Amplitude']
            if not np.isnan(t_amp):
                features[f'{lead_name}_T_Inversion'] = 1 if t_amp < 0 else 0
            else:
                features[f'{lead_name}_T_Inversion'] = np.nan

            #Q-wave Amplitude
            q_peak = avg_amplitude(waves.get("ECG_Q_Peaks", []), ecg_clean)
            features[f'{lead_name}_Q_Amplitude'] = (q_peak - baseline) if not np.isnan(q_peak) else np.nan

            #R-wave Amplitude
            r_peaks_indices = rpeaks.get("ECG_R_Peaks", [])
            r_amp = avg_amplitude(r_peaks_indices, ecg_clean)
            features[f'{lead_name}_R_Amplitude'] = (r_amp - baseline) if not np.isnan(r_amp) else np.nan

            #QRS Duration (ms)
            features[f'{lead_name}_QRS_Duration'] = avg_duration(
                waves.get("ECG_R_Onsets", []),
                waves.get("ECG_R_Offsets", []),
                sampling_rate
            )

            #P Duration (ms)
            features[f'{lead_name}_P_Duration'] = avg_duration(
                waves.get("ECG_P_Onsets", []),
                waves.get("ECG_P_Offsets", []),
                sampling_rate
            )

        target_peaks = rpeaks_lead_ii if rpeaks_lead_ii is not None else rpeaks
        r_indices = target_peaks.get("ECG_R_Peaks", [])

        if len(r_indices) >= 3:
            rr_intervals = np.diff(r_indices) / sampling_rate * 1000
            features['HR_Mean'] = 60000 / np.mean(rr_intervals) if np.mean(rr_intervals) > 0 else np.nan

            try:
                hrv = nk.hrv(target_peaks, sampling_rate=sampling_rate)
                features['HRV_RMSSD']  = hrv['HRV_RMSSD'].values[0]
                features['HRV_MeanNN'] = hrv['HRV_MeanNN'].values[0]
                features['HRV_SDNN']   = hrv['HRV_SDNN'].values[0]
                features['HRV_pNN50']  = hrv['HRV_pNN50'].values[0] if 'HRV_pNN50' in hrv.columns else np.nan
            except:
                features['HRV_RMSSD']  = np.nan
                features['HRV_MeanNN'] = np.nan
                features['HRV_SDNN']   = np.nan
                features['HRV_pNN50']  = np.nan
        else:
            for f in ['HR_Mean', 'HRV_RMSSD', 'HRV_MeanNN', 'HRV_SDNN', 'HRV_pNN50']:
                features[f] = np.nan

        return pd.Series(features)

    except Exception as e:
        print(f"  Error: {filepath} -> {e}")
        return None


#4 - PROCESS DATASET
def process_dataset(df_subset, subset_name):
    """Process all ECG records in a subset and return a feature DataFrame."""
    print(f"\nProcessing {subset_name} ({len(df_subset)} records)...")
    results = []

    for ecg_id, row in tqdm(df_subset.iterrows(), total=len(df_subset)):
        path = row['filename']
        feats = extract_MI_Features(path)

        if feats is not None:
            feats['ecg_id']   = ecg_id
            feats['mi_label'] = row['mi_label']  # Ground truth: MI = 1, Normal = 0.
            results.append(feats)

    if results:
        final_df = pd.DataFrame(results)
        final_df = final_df.set_index('ecg_id')
        return final_df
    else:
        return pd.DataFrame()



if __name__ == "__main__":
    train_mask, val_mask, test_mask = split_data(df_filtered)

    train_df = df_filtered[train_mask].copy()
    val_df   = df_filtered[val_mask].copy()
    test_df  = df_filtered[test_mask].copy()

    print(f"\nSplit sizes -> Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

#All 3 csv files
    X_train = process_dataset(train_df, "Training Set")
    X_train.to_csv('X_train_MI_Features.csv')
    print(f"Train saved: {X_train.shape}")

    X_val = process_dataset(val_df, "Validation Set")
    X_val.to_csv('X_val_MI_Features.csv')
    print(f"Val saved: {X_val.shape}")

    X_test = process_dataset(test_df, "Test Set")
    X_test.to_csv('X_test_MI_Features.csv')
    print(f"Test saved: {X_test.shape}")

#Output
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    feat_cols = [c for c in X_train.columns if c != 'mi_label']
    print(f"Total features: {len(feat_cols)}")
    print(f"Train: {X_train.shape}  |  Val: {X_val.shape}  |  Test: {X_test.shape}")
    print(f"\nFeature list:")
    for col in feat_cols:
        print(f"  - {col}")
