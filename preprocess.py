import pretty_midi
import numpy as np
import zlib
import sys
import scipy
from scipy.sparse import csr_matrix, save_npz, load_npz
import torch
from torch.utils.data import DataLoader


def midi_to_piano_roll(midi_file, fs=100):
    midi_data = pretty_midi.PrettyMIDI(midi_file)
    piano_roll = midi_data.get_piano_roll(fs=fs)
    return piano_roll


def create_dataset(midi_files):
    piano_rolls = []
    for file in midi_files:
        piano_roll = midi_to_piano_roll(file)
        processed_roll = remove_duplicates(piano_roll)
        piano_rolls.append(processed_roll)

    return np.array(piano_rolls, dtype=np.uint16)


def remove_duplicates(piano_roll):
    piano_roll = piano_roll.T

    mask = np.any(np.diff(piano_roll, axis=0) != 0, axis=1)
    mask = np.append(mask, True)
    change_indices = np.where(mask)[0]
    processed_roll = piano_roll[change_indices]
    counts = np.diff(np.concatenate(([-1], change_indices)))
    processed_roll[:, 127] = counts

    return processed_roll



# Example MIDI files (replace with your dataset)
midi_files = ['./MIDI-Unprocessed_Chamber2_MID--AUDIO_09_R3_2018_wav--1.midi']
seq_length = 100
X = create_dataset(midi_files)
sparse_vector = X[0].T

csr_matrix = scipy.sparse.csr_matrix(sparse_vector)

scipy.sparse.save_npz('compressed_matrix.npz', csr_matrix)

# Load the CSR matrix from the file
loaded_matrix = scipy.sparse.load_npz('compressed_matrix.npz')

# To convert back to a dense numpy array if needed
dense_matrix = loaded_matrix.toarray()

print(dense_matrix[:, 1])
