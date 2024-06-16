import matplotlib.pyplot as plt
import numpy as np
import csv
import pandas as pd


#fft from NuRadioReco utilities 

def time2freq(trace, sampling_rate):
    return np.fft.rfft(trace, axis=-1) / sampling_rate * 2 ** 0.5

#effective height plot

def plot_effective_height(data, n, new_figure = True, color = "Purple", linestyle = "-", legend = True, labels = True):

    ts = data["ts"]
    h_the = data["h_the"]

    h_phi = data["h_phi"]
    azimuth_angles = data["azimuth_angles"]
    zenith_angles = data["zenith_angles"]

    nsamples = len(ts)
    fs = ts[1] - ts[0]
    freqs = np.fft.rfftfreq(len(h_phi[0][0]), fs)

    
    #digitized line at 90 degrees zenith 
    freq_dig = []
    spec_dig = []
    path = "dan_90_recent.csv"
    with open(path) as f:
        reader = csv.reader(f)
        for row in reader:
            freq_dig.append(float(row[0]))
            spec_dig.append(float(row[1]))

    #line from Github at 90 degrees zenith 

    plt.figure()
    for i_azimuth_angle, azimuth_angle in enumerate(azimuth_angles):
        for i_zenith_angle, zenith_angle in enumerate(zenith_angles):

            trace_ = np.fft.fftshift(h_the[i_azimuth_angle][i_zenith_angle])
            ts_ = ts - ts[np.argmin(trace_)]

            n_samples_dan = len(ts_)
            sampling_frequency_dan = 1/(ts_[1] - ts_[0])
            spectrum_dan = time2freq(trace_, sampling_frequency_dan)
            frequencies_dan = np.fft.rfftfreq(n_samples_dan, 1 / sampling_frequency_dan)

            if (zenith_angle == 90):
                plt.plot(frequencies_dan, np.abs(spectrum_dan), label = f"Github")
                plt.plot(frequencies_dan, np.abs(spectrum_dan) * 2.33, label = f"Github x2.33")

    
    plt.plot(freq_dig, spec_dig, label = "Digitized")
    plt.legend()
    plt.title("Antenna Effective Height")
    plt.xlabel("Frequency")
    plt.ylabel("VEL Theta")
    plt.savefig("eff_height_zen_comp_90.png")
    plt.close() 

output_file_name_ice = "ice_processed_effective_height.npz"
n_ice = 1.75
r = 7.07106
component_name = "Component"
base_name_ice = "vpol_xfdtd_simulation.xf/ice_output"
data_ice = np.load(output_file_name_ice)
plot_effective_height(data_ice, n_ice, new_figure = False, color = "blue", linestyle = "-", legend = False, labels = False)

