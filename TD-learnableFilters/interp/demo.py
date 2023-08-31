import numpy as np

def hz_to_erb(frequency_hz):
    return 21.4 * np.log10(4.37 * frequency_hz / 1000 + 1)

def erb_to_hz(erb_value):
    return 1000 * ((10 ** (erb_value / 21.4)) - 1) / 4.37

num_values = 64
lowest_freq = 0  # Hz
highest_freq = 8000  # Hz

# Convert frequency to ERB units
lowest_erb = hz_to_erb(lowest_freq)
highest_erb = hz_to_erb(highest_freq)

# Generate uniform values in ERB units
erb_values = np.linspace(lowest_erb, highest_erb, num_values)

# Convert ERB units back to frequency
frequency_values = erb_to_hz(erb_values)

print(frequency_values)
