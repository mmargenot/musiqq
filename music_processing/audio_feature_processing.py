import numpy as np
import pandas as pd
import pydub
import sys
import os
import matplotlib.pyplot as plt
from scipy import stats

from pydub import AudioSegment

def fft_features(x):
    f = np.fft.fft(x)
    f = f[2:int(f.size / 2 + 1)]
    f = abs(f)
    total_power = f.sum()
    f = np.array_split(f, 10)
    return [e.sum() / total_power for e in f]

def moments(x):
    mean = np.mean(x)
    std = np.std(x)
    skewness = stats.skew(x)
    kurtosis = stats.kurtosis(x)
    return [mean, std, skewness, kurtosis]

def other_stuff(song):
    amplitude = song.set_channels(1).get_array_of_samples()
    minute_length = len(amplitude_over_time)/sound_file.frame_rate/60
    
def features(x):
    x = np.array(x)[:(len(x) - len(x) % 1000)] # truncate end so it rounds nicely
    
    f = []
    xs = x
    diff = xs[1:] - xs[:-1]
    f.extend(moments(xs))
    f.extend(moments(diff))
    
    xs = x.reshape(-1, 10).mean(1)
    diff = xs[1:] - xs[:-1]
    f.extend(moments(xs))
    f.extend(moments(diff))
    
    xs = x.reshape(-1, 100).mean(1)
    diff = xs[1:] - xs[:-1]
    f.extend(moments(xs))
    f.extend(moments(diff))
    
    xs = x.reshape(-1, 1000).mean(1)
    diff = xs[1:] - xs[:-1]
    f.extend(moments(xs))
    f.extend(moments(diff))
    
    f.extend(fft_features(x))
    return f

def compute_features(sound):
    amplitude = sound.set_channels(1).get_array_of_samples()
    return features(amplitude)

def generate_samples(sound, sample_length, n_samples):
    start_indices = np.random.choice(np.random.randint(0, len(sound) - sample_length), n_samples, replace=0)
    # create two random samples
    segments = [sound[start_index:start_index+sample_length] for start_index in start_indices]

    return segments


# Load all files into AudioSegments
#all_audio = []
song_stats = {}
song_segments = {}
for path, dirs, files in os.walk(music_folder):
    for f in files:
        #if not (f.endswith('.mp3') or f.endswith('.flac')):
        #    continue
        try:
            sound = AudioSegment.from_file(os.path.join(path, f), format=os.path.splitext(f)[1][1:])
        except pydub.exceptions.CouldntDecodeError:
            print("Couldn't decode {0}".format(f))
            continue
        except OSError:
            print('Uhhhh {0}'.format(f))
            continue
        # Do math on the audio segments
        song_stats[f] = compute_features(sound)
        
        if len(sound) < 60000:
            continue
        segment_a, segment_b = generate_samples(sound)
        song_segments[f] = (compute_features(segment_a), compute_features(segment_b))

all_features = [
    'amp1mean',
    'amp1std',
    'amp1skew',
    'amp1kurt',
    'amp1dmean',
    'amp1dstd',
    'amp1dskew',
    'amp1dkurt', 
    'amp10mean', 
    'amp10std',
    'amp10skew',
    'amp10kurt',
    'amp10dmean',
    'amp10dstd',
    'amp10dskew',
    'amp10dkurt',
    'amp100mean',
    'amp100std',
    'amp100skew',
    'amp100kurt',
    'amp100dmean',
    'amp100dstd',
    'amp100dskew',
    'amp100dkurt',
    'amp1000mean',
    'amp1000std',
    'amp1000skew',
    'amp1000kurt',
    'amp1000dmean',
    'amp1000dstd',
    'amp1000dskew',
    'amp1000dkurt',
    'power1',
    'power2',
    'power3',
    'power4',
    'power5',
    'power6',
    'power7',
    'power8',
    'power9',
    'power10'
]        

data = pd.DataFrame.from_dict(song_stats, orient='index')
data.columns = all_features 
        
data.to_csv('../data/calculated_song_features.csv')
        
# Pull N samples from each song
more_song_segments = {}
samples = 30
for path, dirs, files in os.walk(music_folder):
    for f in files:
        try:
            sound = AudioSegment.from_file(os.path.join(path, f), format=os.path.splitext(f)[1][1:])
        except pydub.exceptions.CouldntDecodeError:
            print("Couldn't decode {0}".format(f))
            continue
        except OSError:
            print('Uhhhh {0}'.format(f))
            continue
        # Do math on the audio segments
        if len(sound) < 60000:
            continue
        segments = generate_samples(sound, 30000, samples)
        more_song_segments[f] = [compute_features(segment) for segment in segments]
    
    
feature_selection_df = pd.DataFrame(columns=all_features+['class'])
class_map = {}
i = 0
j = 0
for key, value in more_song_segments.items():
    class_map[j] = key
    for elt in value:
        feature_selection_df.loc[i] = elt+[j]
        i+=1
    j+=1
    
feature_selection_df.to_csv('../data/sampled_feature_data_with_labels.csv')
