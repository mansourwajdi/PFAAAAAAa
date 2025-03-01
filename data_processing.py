import random

from CQCC.cqcc import cqcc
from python_speech_features import mfcc
import scipy.io.wavfile as wav
import soundfile as sf
import os
import numpy as np
import pickle
import argparse
from notebooks.embedding import extract_cqcc


parser = argparse.ArgumentParser()
parser.add_argument("--data_path", required=True, type=str,
                    help='path to ASVSpoof data directory. For example, LA/ASVspoof2019_LA_train/flac/')
parser.add_argument("--label_path", required=True, type=str,
                    help='path to label file. For example, LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt')
parser.add_argument("--output_path", required=True, type=str,
                    help='path to output pickle file. For example, ./data/train.pkl')
#parser.add_argument("--feature_type", required=True, type=str, help='select the feature type. cqcc or mfcc')
args = parser.parse_args()
path = r"Data/PA/ASVspoof2019_PA_cm_protocols/ASVspoof2019.PA.cm.train.trl.txt"


# Read in labels
filename2label = {}
for line in open(args.label_path):
    line = line.split()
    filename, label = line[1], line[-1]
    filename2label[filename] = label

# Shuffle file processing
all_files = list(os.listdir(args.data_path))
random.shuffle(all_files)  # Shuffle the list of files


feats = []
total_files = len(os.listdir(args.data_path))
print(total_files)
processed_files = 0

for filepath in os.listdir(args.data_path):
    filename = filepath.split('.')[0]
    if filename not in filename2label:  # Skip speaker enrollment stage
        continue
    label = filename2label[filename]
    print("Processing file:", os.path.join(args.data_path, filepath))
    sig, rate = sf.read(os.path.join(args.data_path, filepath))
    feat_cqcc, fmax, fmin = extract_cqcc(sig, rate)
    numframes = feat_cqcc.shape[0]
    winstep = 0.005
    winlen = (len(sig) - winstep * rate * (numframes - 1)) / rate
    feat_mfcc = mfcc(sig, rate, winlen=winlen, winstep=winstep, lowfreq=fmin, highfreq=fmax)
    feats.append((feat_cqcc, feat_mfcc, label))

    # Update progress
    processed_files += 1
    percentage = (processed_files / total_files) * 100
    print("Progress: {:.2f}%".format(percentage))
    if percentage>=7.5:
        break

print("Number of instances:", len(feats))

with open(args.output_path, 'wb') as outfile:
    pickle.dump(feats, outfile)
