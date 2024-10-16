import numpy as np
import pandas as pd
import wfdb
import matplotlib.pyplot as plt
from scipy.stats import f_oneway, ttest_ind
import glob
import os
import neurokit2 as nk
from scipy.signal import firwin, lfilter

""" This piece of code downloads the ECG data, calculates QT intervals and performs the 
anova tests for different genotype groups (AA, AB, BB). The ECG data analysis, i.e., calculating the
QT intervals is ready-made for you. """

""" Almost all code have been provided to me in the course Introduction to Biomedical
Informatics. Lecturers Juha Kesseli and Saana Seppälä. Code is not my own. """

path = '/Users/kalle.keisu/binf_python/project'


# This function loads the ECG data files: .dat, and .hea -files.
# .dat files consists of the ECG voltage data
# .hea files consist of the header information (sampling frequency, leads etc.)
def loadEcg(path):
    dat_files = glob.glob(os.path.join(path, "*.dat"))
    hea_files = glob.glob(os.path.join(path, "*.hea"))
    base_names = set([os.path.splitext(os.path.basename(f))[0] for f in dat_files + hea_files])

    ecg_dict, field_dict, fs_dict = {}, {}, {}

    # Read the signal and metadata for each file. The read file consist of field names (contain, e.g.,
    # the sampling frequency and the lead names), and the actual ecg signal data.
    for i, base_name in enumerate(sorted(base_names), start=1):
        ecg, fields = wfdb.rdsamp(os.path.join(path, base_name))
        patient_key = f'Patient{i}'
        ecg_dict[patient_key] = ecg
        field_dict[patient_key] = fields
        fs_dict[patient_key] = fields['fs']

    return ecg_dict, field_dict, fs_dict


# Function to filter ECG signal with FIR filter
# Change the cutoffs: low = 0.5 and high = 150 (bandpass FIR filtering)
# cutoffs are selected due to report instructions
def filterEcg(key, signal, fs, filter_order=20, low=0.5, high= 150):
    # build a filter
    filter_coeffs = firwin(filter_order, [low, high], fs=fs, pass_zero=False)

    # compare the filtered and the original ecg signals by plotting them
    # Latter: Use Lead II (index 1) and apply the filter.
    leadII = signal[:,1]
    filtered_signal = lfilter(filter_coeffs, [1.0], leadII)

    time = np.arange(leadII.size) / fs  # creating a time vector

    # Plotting and comparing the filtering process among specific patients
    if key in ["Patient1", "Patient2", "Patient3"]:
        plt.figure(figsize=(10, 6))
        plt.plot(time, leadII)
        plt.title(f"{key}: Original ECG Lead II")
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.show()

        plt.figure(figsize=(10, 6))
        plt.plot(time, filtered_signal)
        plt.title(f"{key}: Filtered ECG Lead II")
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.show()

    return filtered_signal


# Function to calculate QT intervals.
    # Process the ECG signal using neurokit2 (detects R-peaks, Q, and T points)
    # ecg_process has many steps:
    # 1) cleaning the ecg with ecg_clean()
    # 2) peak detection with ecg_peaks()
    # 3) HR calculus with signal_rate()
    # 4) signal quality assessment with ecg_quality()
    # 5) QRS elimination with ecg_delineate() and
    # 6) cardiac phase determination with ecg_phase().
def calculateQtIntervals(key, filtered_signal, fs):
    ecg_analysis, _ = nk.ecg_process(filtered_signal, sampling_rate=fs)
    q_points = ecg_analysis['ECG_Q_Peaks'] # This is default output of the ecg_process.
    t_points = ecg_analysis['ECG_T_Offsets'] # This is default output of the ecg_process.
    q_indices = q_points[q_points == 1].index.to_list()
    t_indices = t_points[t_points == 1].index.to_list()

    time = np.arange(filtered_signal.size) / fs

    # calculating QT intervals for all patients and saving them
    qt_intervals = []
    for q, t in zip(q_indices, t_indices):
        if t > q:  # Ensure T point is after Q point for a valid QT interval
            qt_interval = (t - q) / fs  # Convert sample indices to time
            qt_intervals.append(qt_interval)

    # plotting filtered ecg and QT intervals for 3 patients only
    if key in ["Patient1", "Patient2", "Patient3"]:
        plt.figure(figsize=(10, 6)) # Determining a size of the figure
        plt.plot(time, filtered_signal, label="Filtered ECG Lead II")
        plt.title(f'{key} - Filtered ECG with QT Intervals')

        for q, t in zip(q_indices, t_indices):
            if t > q:  # Ensure T point is after Q point for a valid QT interval

                # Plot the QT interval as a red line segment
                plt.plot([q / fs, t / fs],
                         [filtered_signal[q], filtered_signal[t]], color='red', lw=2,
                         label='QT Interval' if len(qt_intervals) == 1 else "")  # Label only the first for legend clarity

        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        plt.close()

    return qt_intervals


# Function to calculate and store average QT interval
def calculateAverageQt(ecg_dict, fs_dict):
    # The average QT intervals for all patients will be stored in average_qt_dict
    average_qt_dict = {}
    for key, ecg_signal in ecg_dict.items():
        fs = fs_dict[key] # corresponding sampling freq. for each signal
        filtered_signal = filterEcg(key, ecg_signal, fs) # This calls the filtering function
        qt_intervals = calculateQtIntervals(key, filtered_signal, fs) # Calculates the intervals based on the filtered data
        average_qt_interval = np.mean(qt_intervals) if qt_intervals else None # Calculates the average QT for each patient.
        average_qt_dict[key] = average_qt_interval

    return average_qt_dict


# Function to load and reshape genotype data
# Reshaping to keep the structure for 7 rows
# New genotype starts every 7th row
def loadAndReshapeGenotype(filepath):
    results = pd.read_csv(filepath, delimiter="\t", header=None)
    selected = results.iloc[:, 1::7]  # Select every 7th column starting from index 1 --> each new genotype AA, AB or BB.
    s_array = selected.values
    reshaped = s_array.reshape(7, -1)  # Automatically calculate columns based on data size, when we want to have
                                       # the original 7 rows.
    return reshaped


# Function to extract QT intervals based on genotype AA, AB or BB
# Function goes through the reshaped data, consisting of 7 rows and 129 columns
# 129 is the number of different genotypes i.e. different patients
# One group to study = same genotype from one row. E.g. all BB genotypes from row 1.
def QtByGenotype(reshaped, average_qt_dict, row):
    patients = list(average_qt_dict.keys())
    genotypes = ["AB", "BB", "AA"] # making a list for looping
    qt_results = {}

    for genotype in genotypes:
        indices = np.where(reshaped[row, :] == genotype)[0]
        qt_results[genotype] = [average_qt_dict[patients[idx]] for idx in indices
                                if average_qt_dict[patients[idx]] is not None]

    return qt_results["AB"], qt_results["BB"], qt_results["AA"]

# Function to perform ANOVA and print results
def AnovaTest(qt_AB, qt_BB, qt_AA):

    if len(qt_AB) != 0 and len(qt_BB) != 0 and qt_AA != 0:
        result = f_oneway(qt_AB, qt_BB, qt_AA)
        print(f"Statistic: {result.statistic:.3f} and p-value: {result.pvalue:.3f}")
    else:
        print("Cannot be calculated :-(")

# Not great function but the code works
def StudentTTest(qt_AB, qt_BB):

    result = ttest_ind(qt_AB, qt_BB)
    print(f"Statistic: {result.statistic:.3f} and p-value: {result.pvalue:.3f}")


# Main processing function
# The main function could be much better but code does work
def main():

    # loading data
    ecg_dict, field_dict, fs_dict = loadEcg(path)

    # calcualte average QT in each patient
    average_qt_dict = calculateAverageQt(ecg_dict, fs_dict)

    abnormal_qt_ids = [] # just for reporting
    abnormal_qt_intervals = [] # just for reporting

    # save the values from 6. value to 15. value into the list (10)
    list_avg_qt = []

    # for plotting the results asked in the instructions
    # list about average QT intervals
    # Latter inner if: list the patients having abnormal QT intervals
    # I chose 0.6 as a threshold even though it is a quite low
    for i, (key, value) in enumerate(average_qt_dict.items()):
        if i in range(5,15):
            list_avg_qt.append(int(value*1000)) # as ms
        if value is not None:  # None result from the Patient 106
            if value >= 0.6:
                abnormal_qt_ids.append(key)
                abnormal_qt_intervals.append(value)


    # Load the genotype data
    genotype_dt = loadAndReshapeGenotype("/Users/kalle.keisu/binf_python/project/GSE55230_FilteredUnixGTMatrix_cut.txt")

    # initialize the lists
    qt_AB = []
    qt_BB = []
    qt_AA = []

    # call QtByGenotype for each variant
    # use index to recognize right observations
    # could be e.g. QtByGenotype(genotype_dt, average_qt_dict, 2) # variant 3
    for i in range(7):
        qt_AB_i, qt_BB_i, qt_AA_i = QtByGenotype(genotype_dt, average_qt_dict, i)
        qt_AB.append(qt_AB_i)
        qt_BB.append(qt_BB_i)
        qt_AA.append(qt_AA_i)

        print(f"Mean (QT) for AB for variant {i+1}: {np.mean(qt_AB_i)*1000:.0f}")
        print(f"Mean (QT) for BB for variant {i+1}: {np.mean(qt_BB_i)*1000:.0f}")
        print(f"Mean (QT) for AA for variant {i+1}: {np.mean(qt_AA_i)*1000:.0f}")

    # Perform statistical analysis tests (one-way ANOVA and Student-T test)
    # for each variant
    for i in range(7):
        if i != 4:  # Perform Student-T test for fifth variant
            AnovaTest(qt_AB[i], qt_BB[i], qt_AA[i])
        else:
            StudentTTest(qt_AB[i], qt_BB[i])


# Run the main function
if __name__ == "__main__":
    main()
