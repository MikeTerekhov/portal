import io
import base64
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from flask import Flask, render_template, request, jsonify

def preprocess(csv1, csv2):

  df_1 = pd.read_csv(csv1)

  df_toGetMerged = pd.read_csv(csv2)


  # rename both columns to subject_id instead of participant
  df_toGetMerged = df_toGetMerged.rename(columns = {"Participant ID  {participantNumber}_{studyID}    Study ID: 300    Subject Number Ranges:  UDelaware: 001-099  Mayo Clinic: 100-199   CS Fresno: 200-299   USC: 335-399  CHLA: 400-499  JHU: 500-599  UTSWMC: 600-699  NMD: 700-799  HSS: 800-899": "subject_id",
                                                    "Dominant LEG": "dominant_leg"}) # Rename 'Dominant LEG'

  # Convert 'subject_id' columns to string type for merging
  df_1['subject_id'] = df_1['subject_id'].astype(str)
  df_toGetMerged['subject_id'] = df_toGetMerged['subject_id'].astype(str)


  # starting on row 100 until row 220,
  # take the current subject_id, if "_" is in it, split on "_", make the label the 0th index element (0 based index)
  for index in range(100, 221):
      subject_id = df_toGetMerged.loc[index, 'subject_id']
      if isinstance(subject_id, str) and "_" in subject_id:
          df_toGetMerged.loc[index, 'subject_id'] = subject_id.split("_")[0]

  # on same subject ID, I want to combine the rows from the two dataframes into a new one

  merged_df = pd.merge(df_1, df_toGetMerged, on='subject_id', how='left')

  # merged_df.to_csv("All_merged_data_300", index = False )


  controls = merged_df['subject_id'].astype(str).str.startswith('7') | (merged_df['subject_id'].astype(str).astype(int) >= 900)

  # Create the two dataframes
  df_controls = merged_df[controls].copy()

  df_controls = df_controls[(df_controls['force_symmetry'].notna()) & (df_controls['force_symmetry'] != 0)]

  # Filter for 'force_symmetry' values greater than -20 -> outliers

  df_controls_filtered = df_controls[df_controls['force_symmetry'] > -20].copy()

  df_controls_filtered = df_controls_filtered[df_controls_filtered['left_force'] != 0]
  df_controls_filtered = df_controls_filtered[df_controls_filtered['right_force'] != 0]



  ### Create dominant symmetry values ###


  df_controls_filtered['dominance_symmetry'] = np.where(
      df_controls_filtered['dominant_leg'] == 1,
      np.where(df_controls_filtered['right_force'] != 0,
              (df_controls_filtered['right_force'] - df_controls_filtered['left_force']) / df_controls_filtered['right_force'],
              np.nan),
      np.where(df_controls_filtered['left_force'] != 0,
              (df_controls_filtered['left_force'] - df_controls_filtered['right_force']) / df_controls_filtered['left_force'],
              np.nan)
  )

### Non dominant ###
  df_controls_filtered['nondominance_symmetry'] = np.where(
      df_controls_filtered['dominant_leg'] == 1,
      np.where(df_controls_filtered['left_force'] != 0,
              (df_controls_filtered['left_force'] - df_controls_filtered['right_force']) / df_controls_filtered['left_force'],
              np.nan),
          np.where(df_controls_filtered['right_force'] != 0,
              (df_controls_filtered['right_force'] - df_controls_filtered['left_force']) / df_controls_filtered['right_force'],
              np.nan),
  )



  df_controls_filtered['right_symmetry'] = np.where(df_controls_filtered['right_force'] != 0,
              (df_controls_filtered['right_force'] - df_controls_filtered['left_force']) / df_controls_filtered['right_force'],
              np.nan)



  df_controls_filtered['left_symmetry'] =  np.where(df_controls_filtered['right_force'] != 0,
              (df_controls_filtered['left_force'] - df_controls_filtered['right_force']) / df_controls_filtered['left_force'],
              np.nan)

  ### Convert into Percents ###
  df_controls_filtered['dominance_symmetry'] = df_controls_filtered['dominance_symmetry'].copy()*100

  # df_controls_filtered['left_symmetry'] = df_controls_filtered['left_symmetry'].copy()*100

  df_controls_filtered['right_symmetry'] = df_controls_filtered['right_symmetry'].copy()*100

  return df_controls_filtered

# -------------

dataframe_controls_filter = preprocess('./300_natalie_symmetry_report_2025-10-20.csv', './AssessmentOfNeuromus_DATA_LABELS_2025-09-17_1911.csv' )

print(dataframe_controls_filter)
