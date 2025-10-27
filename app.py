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

def preprocessInjuried(csv1, csv2):

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


  controls = merged_df['subject_id'].astype(str).str.startswith('7') | (merged_df['subject_id'].astype(str).astype(int) <= 900)

  # Create the two dataframes
  df_controls = merged_df[controls].copy()

  df_controls = df_controls[(df_controls['force_symmetry'].notna()) & (df_controls['force_symmetry'] != 0)]

  inverse_controls = ~(
    merged_df['subject_id'].astype(str).str.startswith('7') |
    (merged_df['subject_id'].astype(str).astype(int) >= 900)
  )

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

def plotFMAttributes(dataframe, category, sex, bins=30):

  # get the sex
  dataframe = dataframe[dataframe['Participant Sex'] == sex].copy()


  # sanitize the data

  data = (
        pd.to_numeric(dataframe[category], errors='coerce')
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .values
  )


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
  popt, pcov = curve_fit(gaussian_func, bin_centers, counts, p0=p0)

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

  plt.xlabel('Value')
  plt.ylabel('Counts')
  plt.title(f'Gaussian Fit to {sex} {category}')
  plt.legend()
  plt.grid(axis='y', linestyle='--', alpha=0.7)
  plt.savefig('gaussian_fit.png')
  plt.show()

  return fit_amp, fit_mu, fit_sig

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
            processed_data = preprocess(symmetry_file, assessment_file)
        except Exception as e:
            return jsonify({'error': f'Data preprocessing failed: {str(e)}'}), 400

        # Generate Gaussian fit and plot
        try:
            amp, mu, sigma = plotFMAttributes(processed_data, category, sex)
        except Exception as e:
            return jsonify({'error': f'Gaussian fit failed: {str(e)}'}), 400

        # Capture the current Matplotlib figure in memory
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_data = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        
        return jsonify({
            'plot': img_data,
            'stats': {
                'amplitude': amp,
                'mean': mu,
                'std_dev': sigma
            }
        })
    
    except Exception as e:
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500


if __name__ == '__main__':
    app.run(debug=True)
