from pathlib import Path
from random import shuffle
from miditok.data_augmentation import augment_dataset
from miditok.pytorch_data.split_utils import split_files_for_training
from miditok.pytorch_data import DatasetMIDI, DataCollator
from miditok import REMI, TokenizerConfig
from torch.utils.data import DataLoader

import constants


def train_tokenizer(raw_data_path, tokenizer_path):
    config = TokenizerConfig(num_velocities=16, use_chords=True, use_programs=True)
    tokenizer = REMI(config)
    
    # Train the tokenizer with Byte Pair Encoding (BPE)
    midi_paths = list(Path(raw_data_path).glob("**/*.midi"))
    tokenizer.train(vocab_size=30000, files_paths=midi_paths)
    tokenizer.save_params(Path(tokenizer_path))
    return tokenizer


def create_dataset(tokenizer, raw_data_path, tokenizer_path):
    # Split the dataset into train/valid/test subsets, with 15% of the data for each of the two latter
    midi_paths = list(Path(raw_data_path).glob("**/*.midi"))
    total_num_files = len(midi_paths)
    num_files_valid = round(total_num_files * VALID_RATIO)
    num_files_test = round(total_num_files * TEST_RATIO)
    shuffle(midi_paths)
    midi_paths_valid = midi_paths[:num_files_valid]
    midi_paths_test = midi_paths[num_files_valid:num_files_valid + num_files_test]
    midi_paths_train = midi_paths[num_files_valid + num_files_test:]

    tokenizer = REMI(tokenizer_path)
    
    # Chunk MIDIs and perform data augmentation on each subset independently
    for files_paths, subset_name in (
        (midi_paths_train, "train"), (midi_paths_valid, "valid"), (midi_paths_test, "test")
    ):
    
        # Split the MIDIs into chunks of sizes approximately about 1024 tokens
        subset_chunks_dir = Path(f"dataset_{subset_name}")
        split_files_for_training(
            files_paths=files_paths,
            tokenizer=tokenizer,
            save_dir=subset_chunks_dir,
            max_seq_len=BLOCK_SIZE,
            num_overlap_bars=2,
        )
    
        # Perform data augmentation
        augment_dataset(
            subset_chunks_dir,
            pitch_offsets=[-12, 12],
            velocity_offsets=[-4, 4],
            duration_offsets=[-0.5, 0.5],
        )


def get_dataloader(dataset_path, tokenizer_path):
    tokenizer = REMI(params=tokenizer_path)
    midi_paths = list(Path(dataset_path).glob("**/*.midi"))
    dataset = DatasetMIDI(
        files_paths=midi_paths,
        tokenizer=tokenizer,
        max_seq_len=constants.BLOCK_SIZE,
        bos_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer["BOS_None"],
    )
    collator = DataCollator(tokenizer.pad_token_id)
    data_loader = DataLoader(dataset=dataset, collate_fn=collator, batch_size=constants.BATCH_SIZE)

    return data_loader
