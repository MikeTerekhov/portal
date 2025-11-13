import io
import base64
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

def preprocess(csv1, csv2):

  print("pre 1")

  df_1 = pd.read_csv(csv1)

  df_toGetMerged = pd.read_csv(csv2)


  # rename both columns to subject_id instead of participant
  df_toGetMerged = df_toGetMerged.rename(columns = {"Participant ID  {participantNumber}_{studyID}    Study ID: 300    Subject Number Ranges:  UDelaware: 001-099  Mayo Clinic: 100-199   CS Fresno: 200-299   USC: 335-399  CHLA: 400-499  JHU: 500-599  UTSWMC: 600-699  NMD: 700-799  HSS: 800-899": "subject_id",
                                                    "Dominant LEG": "dominant_leg"})

  # df_toGetMerged = df_toGetMerged.rename(
  #       columns={
  #           "participant_id": "subject_id",
  #           "Dominant LEG": "dominant_leg",
  #           "participant_sex": "Participant Sex",
  #       }
  #   )


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

def preprocessInjuried(csv1, csv2):
    """
    csv1: symmetry report
    csv2: complementary data // assessment
    returns: filtered injured dataframe with symmetry columns
    """
    print("pre injure")

    # 1) read both
    df_1 = pd.read_csv(csv1)      # symmetry report
    df_meta = pd.read_csv(csv2)   # complementary data

    # 2) normalize column names
    df_1.columns = [c.strip() for c in df_1.columns]
    df_meta.columns = [c.strip() for c in df_meta.columns]

    # 3) rename in meta so we can merge on the same key
    df_meta = df_meta.rename(
        columns={
            "participant_id": "subject_id",
            "Dominant LEG": "dominant_leg",
            "participant_sex": "Participant Sex",
        }
    )
    # 4) normalize IDs in BOTH (strip, drop underscore parts, drop leading zeros)
    df_meta["subject_id"] = (
        df_meta["subject_id"]
        .astype(str)
        .str.strip()
        .str.split("_").str[0]
        .str.lstrip("0")
    )
    df_1["subject_id"] = (
        df_1["subject_id"]
        .astype(str)
        .str.strip()
        .str.lstrip("0")
    )

    # 5) collapse the meta file to ONE ROW PER SUBJECT
    # builds the agg dict dynamically so we don't drop injured_leg_acl if it's there
    agg_dict = {
        "consent_acl": "max",
        "reason_acl_test": "first",
        "dominant_leg": "first",
        "Participant Sex": "first",
    }
    # if assessment csv has injured_leg_acl, keep it too
    if "injured_leg_acl" in df_meta.columns:
        agg_dict["injured_leg_acl"] = "first"

    meta_per_subject = (
        df_meta
        .groupby("subject_id", as_index=False)
        .agg(agg_dict)
    )

    # 6) merge symmetry report with per-subject meta
    merged_df = pd.merge(df_1, meta_per_subject, on="subject_id", how="left")

    # 7) keep only consented subjects -> injured
    if "consent_acl" not in merged_df.columns:
        raise KeyError("'consent_acl' not found after merge. Check second CSV columns.")
    df_injuried = merged_df[merged_df["consent_acl"] == 1]

    # 8) symmetry-validity filters
    df_injuried = df_injuried[
        (df_injuried["force_symmetry"].notna()) &
        (df_injuried["force_symmetry"] != 0)
    ]

    df_injuried_filtered = df_injuried[df_injuried["force_symmetry"] > -20].copy()
    df_injuried_filtered = df_injuried_filtered[df_injuried_filtered["left_force"] != 0]
    df_injuried_filtered = df_injuried_filtered[df_injuried_filtered["right_force"] != 0]

    # 9) dominance_symmetry
    df_injuried_filtered["dominance_symmetry"] = np.where(
        df_injuried_filtered["dominant_leg"] == 1,
        np.where(
            df_injuried_filtered["right_force"] != 0,
            (df_injuried_filtered["right_force"] - df_injuried_filtered["left_force"])
            / df_injuried_filtered["right_force"],
            np.nan,
        ),
        np.where(
            df_injuried_filtered["left_force"] != 0,
            (df_injuried_filtered["left_force"] - df_injuried_filtered["right_force"])
            / df_injuried_filtered["left_force"],
            np.nan,
        ),
    )

    # 10) non-dominance symmetry
    df_injuried_filtered["nondominance_symmetry"] = np.where(
        df_injuried_filtered["dominant_leg"] == 1,
        np.where(
            df_injuried_filtered["left_force"] != 0,
            (df_injuried_filtered["left_force"] - df_injuried_filtered["right_force"])
            / df_injuried_filtered["left_force"],
            np.nan,
        ),
        np.where(
            df_injuried_filtered["right_force"] != 0,
            (df_injuried_filtered["right_force"] - df_injuried_filtered["left_force"])
            / df_injuried_filtered["right_force"],
            np.nan,
        ),
    )

    # 11) right / left symmetry
    df_injuried_filtered["right_symmetry"] = np.where(
        df_injuried_filtered["right_force"] != 0,
        (df_injuried_filtered["right_force"] - df_injuried_filtered["left_force"])
        / df_injuried_filtered["right_force"],
        np.nan,
    )
    df_injuried_filtered["left_symmetry"] = np.where(
        df_injuried_filtered["right_force"] != 0,
        (df_injuried_filtered["left_force"] - df_injuried_filtered["right_force"])
        / df_injuried_filtered["right_force"],
        np.nan,
    )

    ### injury_symmetry calc (injury-noninjury)/injury

    df_injuried_filtered["injury_symmetry"] = np.where(
        df_injuried_filtered["injured_leg_acl"] == 1,  # left injured
        np.where(
            df_injuried_filtered["left_force"] != 0,
            (df_injuried_filtered["left_force"] - df_injuried_filtered["right_force"])
            / df_injuried_filtered["left_force"],
            np.nan,
        ),
        np.where(  # right injured
            df_injuried_filtered["right_force"] != 0,
            (df_injuried_filtered["right_force"] - df_injuried_filtered["left_force"])
            / df_injuried_filtered["right_force"],
            np.nan,
        ),
    )

    ### non injury_symmetry calc (noninjury-injury)/noninjury

    df_injuried_filtered["noninjury_symmetry"] = np.where(
        df_injuried_filtered["injured_leg_acl"] == 2,  # right injured
        np.where(
            df_injuried_filtered["left_force"] != 0,
            (df_injuried_filtered["left_force"] - df_injuried_filtered["right_force"])
            / df_injuried_filtered["left_force"],
            np.nan,
        ),
        np.where(  # right injured
            df_injuried_filtered["right_force"] != 0,
            (df_injuried_filtered["right_force"] - df_injuried_filtered["left_force"])
            / df_injuried_filtered["right_force"],
            np.nan,
        ),
    )

    # compute explicit injury and non-injury forces per row and new_symm = (injury / noninjury) * 100
    injury_force = pd.Series(np.nan, index=df_injuried_filtered.index)
    noninjury_force = pd.Series(np.nan, index=df_injuried_filtered.index)
    mask_left = df_injuried_filtered["injured_leg_acl"] == 1
    mask_right = df_injuried_filtered["injured_leg_acl"] == 2

    # left injured -> injury = left, noninjury = right
    injury_force.loc[mask_left] = df_injuried_filtered.loc[mask_left, "left_force"]
    noninjury_force.loc[mask_left] = df_injuried_filtered.loc[mask_left, "right_force"]
    # right injured -> injury = right, noninjury = left
    injury_force.loc[mask_right] = df_injuried_filtered.loc[mask_right, "right_force"]
    noninjury_force.loc[mask_right] = df_injuried_filtered.loc[mask_right, "left_force"]

    # avoid division by zero
    noninjury_nz = noninjury_force.replace({0: np.nan})

    df_injuried_filtered["injury_force"] = injury_force
    df_injuried_filtered["noninjury_force"] = noninjury_force
    df_injuried_filtered["new_symm"] = (injury_force / noninjury_nz) * 100

    print("----------")
    print(df_injuried_filtered["new_symm"])
    print("----------")

    # 13) convert some to percents
    df_injuried_filtered["dominance_symmetry"] = df_injuried_filtered["dominance_symmetry"] * 100
    df_injuried_filtered["right_symmetry"] = df_injuried_filtered["right_symmetry"] * 100
    df_injuried_filtered["injury_symmetry"] = df_injuried_filtered["injury_symmetry"] * 100

    return df_injuried_filtered

def plotFMAttributes2(dataframe, category, sex, bins=30):

  # Map string sex input to numerical values
  if isinstance(sex, str):
      sex = sex.lower()
      if sex in ("female", "f"):
          sex_numeric = 1
      elif sex in ("male", "m"):
          sex_numeric = 2
      else:
          print(f"Warning: Invalid sex value '{sex}'. No data to plot.")
          return 0, np.nan, np.nan
  elif sex in (1, 2):
      sex_numeric = sex
  else:
      print(f"Warning: Invalid sex value '{sex}'. No data to plot.")
      return 0, np.nan, np.nan


  # Filter by numerical sex value (1 for Female, 2 for Male)
  dataframe = dataframe[dataframe['Participant Sex'] == sex_numeric].copy()

  sex_count = len(dataframe)


  # sanitize the data

  data = (
        pd.to_numeric(dataframe[category], errors='coerce')
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .values
  )

  if len(data) == 0:
      print(f"No data available for {sex} in category {category}.")
      return 0, np.nan, np.nan

  # empirical (raw data) statistics
  data_mean = np.mean(data)
  data_std = np.std(data)

  # --- 2. Create the Histogram ---
  # We need the y-values (counts) and the x-values (bin centers)
  counts, bin_edges = np.histogram(data, bins=bins)

  # Calculate the center of each bin for the x-data
  bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

  # --- 3. Define the Gaussian Function ---
  # This is the function we want to fit.
  def gaussian_func(x, amp, mu, sig):
      return amp * np.exp(- (x - mu)**2 / (2 * sig**2))

  # --- 4. Provide Initial Guesses ---
  # curve_fit works better with good starting points.
  guess_amp = np.max(counts)
  guess_mu = np.mean(data)
  guess_sig = np.std(data)
  p0 = [guess_amp, guess_mu, guess_sig]

  # --- 5. Run the Fit ---
  # Fit 'gaussian_func' to the (bin_centers, counts) data
  try:
    popt, pcov = curve_fit(gaussian_func, bin_centers, counts, p0=p0)
  except RuntimeError as e:
    print(f"Could not fit curve: {e}")
    return 0, np.nan, np.nan


  # 'popt' contains the optimal parameters: [amp, mu, sig]
  fit_amp, fit_mu, fit_sig = popt

  print("Best Fit Parameters (Method 1):")
  print(f"  Amplitude = {fit_amp:.2f}")
  print(f"  Mean (mu) = {fit_mu:.2f}")
  print(f"  Std Dev (sigma) = {fit_sig:.2f}")

  # --- 6. Plot the Results ---
  plt.figure(figsize=(10, 6))

  # Plot the histogram
  bin_width = bin_edges[1] - bin_edges[0]
  plt.bar(bin_centers, counts, width=bin_width, color='skyblue', alpha=0.7, label='Histogram Bins')

  # Plot the fitted curve
  x_fit = np.linspace(bin_edges[0], bin_edges[-1], 200)
  y_fit = gaussian_func(x_fit, *popt)
  plt.plot(x_fit, y_fit, 'r-', linewidth=2, label='Best Fit Gaussian')

  # White = very high symmetry (within ±0.5σ)
  plt.axvspan(fit_mu - 0.5 * fit_sig, fit_mu + 0.5 * fit_sig, alpha=0.4, color='blue', label='High Symmetry (±0.5σ)')

  # Green = typical range (within ±1σ)
  plt.axvspan(fit_mu - fit_sig, fit_mu + fit_sig, alpha=0.15, color='green', label='Typical (±1σ)')

  # Orange = borderline (1σ – 1.5σ)
  plt.axvspan(fit_mu - fit_sig * 1.5, fit_mu - fit_sig, alpha=0.15, color='orange')
  plt.axvspan(fit_mu + fit_sig, fit_mu + fit_sig * 1.5, alpha=0.15, color='orange', label='Borderline (1–1.5σ)')

  # Red = atypical (beyond ±1.5σ)
  plt.axvspan(bin_edges[0], fit_mu - fit_sig * 1.5, alpha=0.10, color='red')
  plt.axvspan(fit_mu + fit_sig * 1.5, bin_edges[-1], alpha=0.10, color='red', label='Atypical (>1.5σ)')

  plt.xlabel('Value')
  plt.ylabel('Counts')
  plt.title(f'Gaussian Fit to {sex} {category}')
  plt.legend()
  plt.grid(axis='y', linestyle='--', alpha=0.7)
  plt.savefig('gaussian_fit.png')
  plt.show()

  return {
        'data_mean': data_mean,
        'data_std': data_std,
        'fit_mean': fit_mu,
        'fit_std': fit_sig,
        'fit_amplitude': fit_amp,
        'sex_count': sex_count
    }

def plotFMAttributes(dataframe, category, sex, bins=30):

    # Filter by sex
    dataframe = dataframe[dataframe['Participant Sex'] == sex].copy()
    sex_count = len(dataframe)

    # Extract numeric clean data
    data = (
        pd.to_numeric(dataframe[category], errors='coerce')
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .values
    )

    # empirical (raw data) statistics
    data_mean = np.mean(data)
    data_std = np.std(data)

    # Create histogram
    counts, bin_edges = np.histogram(data, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    # Gaussian model
    def gaussian_func(x, amp, mu, sig):
        return amp * np.exp(- (x - mu)**2 / (2 * sig**2))

    # Initial guesses
    p0 = [np.max(counts), data_mean, data_std]

    # Fit
    popt, pcov = curve_fit(gaussian_func, bin_centers, counts, p0=p0)
    fit_amp, fit_mu, fit_sig = popt

    print("Empirical Data Stats:")
    print(f"  Mean = {data_mean:.2f}")
    print(f"  Std Dev = {data_std:.2f}")

    print("Gaussian Fit Stats:")
    print(f"  Fit Mean (mu) = {fit_mu:.2f}")
    print(f"  Fit Std Dev (sigma) = {fit_sig:.2f}")

    # Plot
    plt.figure(figsize=(10, 6))
    bin_width = bin_edges[1] - bin_edges[0]
    plt.bar(bin_centers, counts, width=bin_width, color='skyblue', alpha=0.7, label='Histogram Bins')

    x_fit = np.linspace(bin_edges[0], bin_edges[-1], 200)
    y_fit = gaussian_func(x_fit, *popt)
    plt.plot(x_fit, y_fit, 'r-', linewidth=2, label='Best Fit Gaussian')

    # White = very high symmetry (within ±0.5σ)
    plt.axvspan(fit_mu - 0.5 * fit_sig, fit_mu + 0.5 * fit_sig, alpha=0.4, color='blue', label='High Symmetry (±0.5σ)')

    # Green = typical range (within ±1σ)
    plt.axvspan(fit_mu - fit_sig, fit_mu + fit_sig, alpha=0.15, color='green', label='Typical (±1σ)')

    # Orange = borderline (1σ – 1.5σ)
    plt.axvspan(fit_mu - fit_sig * 1.5, fit_mu - fit_sig, alpha=0.15, color='orange')
    plt.axvspan(fit_mu + fit_sig, fit_mu + fit_sig * 1.5, alpha=0.15, color='orange', label='Borderline (1–1.5σ)')

    # Red = atypical (beyond ±1.5σ)
    plt.axvspan(bin_edges[0], fit_mu - fit_sig * 1.5, alpha=0.10, color='red')
    plt.axvspan(fit_mu + fit_sig * 1.5, bin_edges[-1], alpha=0.10, color='red', label='Atypical (>1.5σ)')

    plt.xlabel('Value')
    plt.ylabel('Counts')
    plt.title(f'Gaussian Fit to {sex} {category}')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('gaussian_fit.png')
    plt.show()

    # return both sets of values
    return {
        'data_mean': data_mean,
        'data_std': data_std,
        'fit_mean': fit_mu,
        'fit_std': fit_sig,
        'fit_amplitude': fit_amp,
        'sex_count': sex_count
    }

@app.route('/plot-gaussian', methods=['POST'])
def plot_gaussian():
    try:
        # Get uploaded files
        symmetry_file = request.files.get('symmetry_file')
        assessment_file = request.files.get('assessment_file')
        
        # Get parameters
        category = request.form.get('category', 'dominance_symmetry')
        sex = request.form.get('sex', 'Male')
        color = request.form.get('color', 'blue')

        if not symmetry_file or not assessment_file:
            return jsonify({'error': 'Both symmetry and assessment files are required'}), 400
        
        # Process data
        try:
            if category == "dominance_symmetry" or category == "right_symmetry":
                processed_data = preprocess(symmetry_file, assessment_file)
            else:
                processed_data = preprocessInjuried(symmetry_file, assessment_file)
        except Exception as e:
            return jsonify({'error': f'Data preprocessing failed: {str(e)}'}), 400
       
        if category == 'injured_3':
            processed_data = processed_data[processed_data['reason_acl_test'] == 1]
            category = 'injury_symmetry'
            result = plotFMAttributes2(processed_data, category, sex)
        elif category == 'injured_6':
            processed_data = processed_data[processed_data['reason_acl_test'] == 2]
            category = 'injury_symmetry'
            result = plotFMAttributes2(processed_data, category, sex)
        elif category == 'injured_RTS':
            processed_data = processed_data[processed_data['reason_acl_test'] == 3]
            category = 'injury_symmetry'
            result = plotFMAttributes2(processed_data, category, sex)
        elif category == 'new_3':
            processed_data = processed_data[processed_data['reason_acl_test'] == 1]
            category = 'new_symm'
            result = plotFMAttributes2(processed_data, category, sex)
        elif category == 'new_6':
            processed_data = processed_data[processed_data['reason_acl_test'] == 2]
            category = 'new_symm'
            result = plotFMAttributes2(processed_data, category, sex)
        elif category == 'new_RTS':
            processed_data = processed_data[processed_data['reason_acl_test'] == 3]
            category = 'new_symm'
            result = plotFMAttributes2(processed_data, category, sex)
        # NOT INJURED
        else : 
            processed_data = processed_data
            result = plotFMAttributes(processed_data, category, sex)


        print(category)
        print(sex)

        # Capture the current Matplotlib figure in memory
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_data = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()

        data_mean     = result['data_mean']
        data_std      = result['data_std']
        fit_mean      = result['fit_mean']
        fit_std       = result['fit_std']
        fit_amplitude = result['fit_amplitude']
        sex_count    = result['sex_count']
        
        return jsonify({
            'plot': img_data,
            'stats': {
                'data_mean': data_mean,
                'data_std': data_std,
                'fit_mean': fit_mean,
                'fit_std': fit_std,
                'fit_amplitude': fit_amplitude,
                'sample_size': len(processed_data),
                'sex_count': sex_count
            }
        })
    
    except Exception as e:
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500


if __name__ == '__main__':
    app.run(debug=True)
