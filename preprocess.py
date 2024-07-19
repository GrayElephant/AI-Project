import pretty_midi
import numpy as np
import zlib
import sys
import scipy
from scipy.sparse import csr_matrix, save_npz, load_npz
import torch
from torch.utils.data import DataLoader
"""
arr = np.array([[1, 2, 0], [1, 2, 0], [1, 2, 0], [1, 3, 0], [1, 3, 0], [1, 2, 0], [1, 2, 0]])
mask = np.any(np.diff(arr, axis=0) != 0, axis=1)
mask = np.append(mask, True)
change_indices = np.where(mask)[0]
unique_rows = arr[change_indices]
counts = np.diff(np.concatenate(([-1], change_indices)))
unique_rows[:, -1] = counts
print(change_indices)
print(unique_rows)
print(np.diff(np.concatenate(([-1], change_indices))))
"""


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


"""
def create_dataset(midi_files, seq_length):
    sequences = []
    next_notes = []
    for file in midi_files:
        piano_roll = midi_to_piano_roll(file)
        for i in range(len(piano_roll[0]) - seq_length):
            sequences.append(piano_roll[:, i:i+seq_length])
            next_notes.append(piano_roll[:, i+seq_length])
    return np.array(sequences), np.array(next_notes)
"""


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
"""
# Save the CSR matrix to a file
scipy.sparse.save_npz('compressed_matrix.npz', csr_matrix)

# Load the CSR matrix from the file
loaded_matrix = scipy.sparse.load_npz('compressed_matrix.npz')
"""
"""
piano_roll = X[0].T
mask = np.any(np.diff(piano_roll, axis=0) != 0, axis=1)
mask = np.append(mask, True)
change_indices = np.where(mask)[0]
processed_roll = piano_roll[change_indices]
counts = np.diff(np.concatenate(([-1], change_indices)))
processed_roll[:, 127] = counts
print(f"counts: {counts}")
print(processed_roll.shape)
print(piano_roll.shape)
print(processed_roll[1, :])
"""
"""
X: This variable contains sequences of piano rolls extracted from MIDI files. Each element in X represents a sequence of notes from the piano roll, formatted as a NumPy array. The dimensions of each array in X are (128, seq_length), where:

128 corresponds to the number of pitches (notes) in MIDI (from MIDI note number 0 to 127).
seq_length is the length of each sequence of notes, specified as 100 in your example.
"""

"""
y: This variable contains the corresponding next notes following each sequence in X. Similar to X, each element in y is a piano roll sequence, but it represents the next set of notes that follow the sequence in X. Therefore, the dimensions of each array in y are also (128, seq_length).
"""
