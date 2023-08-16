"""
Download, preprocess and serve the TinyStories dataset as a DataLoader.
"""

import argparse
import glob
import json
import os
import random
from typing import List
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
import requests
import torch
import torch.distributed as dist
from tqdm import tqdm




from tokenizer import Tokenizer

DATA_CACHE_DIR = "data"
DATA_URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories_all_data.tar.gz"
FILE_NAME = "TinyStories_all_data.tar.gz"
UNPACKED_DIRNAME = "TinyStories_all_data"

def download_file(url: str, fname: str, chunk_size=1024):
    """Helper function to download a file from a given url"""
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get("content-length", 0))
    with open(fname, "wb") as file, tqdm(
        desc=fname,
        total=total,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)



def download(data_url=DATA_URL, cache_dir=DATA_CACHE_DIR, filename=FILE_NAME, unpacked_dirname=UNPACKED_DIRNAME):
    """
    Downloads and unpacks a dataset.
    
    Args:
    - data_url: URL to download the dataset from.
    - cache_dir: Directory where the dataset will be downloaded.
    - filename: Name of the file that will be downloaded.
    - unpacked_dirname: Name of the directory where the dataset will be unpacked.
    """
    os.makedirs(cache_dir, exist_ok=True)
    data_filename = os.path.join(cache_dir, filename)
    
    # download the dataset, unless it's already downloaded
    if not os.path.exists(data_filename):
        print(f"Downloading {data_url} to {data_filename}...")
        download_file(data_url, data_filename)  # Assuming download_file is available in your code
    else:
        print(f"{data_filename} already exists, skipping download...")
    
    # unpack the file (assuming it's a tar.gz) into the specified directory
    data_dir = os.path.join(cache_dir, unpacked_dirname)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)
        print(f"Unpacking {data_filename}...")
        os.system(f"tar -xzf {data_filename} -C {data_dir}")
    else:
        print(f"{data_dir} already exists, skipping unpacking...")
    
    # print a single example just for debugging and such
    shard_filenames = sorted(glob.glob(os.path.join(data_dir, "*.json")))
    with open(shard_filenames[0], "r") as f:
        data = json.load(f)
    print("Download done.")
    print(f"Number of shards: {len(shard_filenames)}")
    print(f"Example story:\n{data[0]}")


def train_vocab(vocab_size):
    """
    Trains a custom sentencepiece tokenizer on the TinyStories dataset.
    The custom tokenizer files will be saved in DATA_CACHE_DIR/tok{N} directories,
    where N is the vocab size. This is also where the pretok .bin files will go.
    """
    assert vocab_size > 0, "Vocab size must be positive"

    # output file prefix path for sentencepiece
    prefix = os.path.join(DATA_CACHE_DIR, f"tok{vocab_size}")

    # how many shards we'll use for vocab training, kept low for efficiency
    num_shards = 10

    # 1) export a large chunk of text as a single text file tiny.txt
    tiny_file = os.path.join(DATA_CACHE_DIR, "tiny.txt")
    data_dir = os.path.join(DATA_CACHE_DIR, "TinyStories_all_data")
    shard_filenames = sorted(glob.glob(os.path.join(data_dir, "*.json")))

    print(f"Writing temporary file {tiny_file} with {num_shards} shards...")
    with open(tiny_file, "w") as of:
        for shard in tqdm(shard_filenames[:num_shards]):
            with open(shard, "r") as f:
                data = json.load(f)
            for example in data:
                text = example["story"]
                text = text.strip()
                of.write(text + "\n")
    print(f"Size is: {os.path.getsize(tiny_file) / 1024 / 1024:.2f} MB")

    # 2) run the train_vocab.sh script that trains the sentencepiece model
    print("Will now train the vocab with:")
    cmd = f"bash train_vocab.sh {tiny_file} {prefix} {vocab_size}"
    print(cmd)
    print("OK? [y/N] ")
    dec = input()
    if dec.lower() != "y":
        print("Exiting...")
        return
    os.system(cmd)

    # 3) optional cleanup, ask the user if they'd like to delete tiny.txt
    dec = input(f"Delete the temporary file {tiny_file}? [y/N] ")
    if dec.lower() == "y":
        os.remove(tiny_file)
        print(f"Deleted {tiny_file}")

    print(f"Trained tokenizer is in {prefix}.model")
    print("Done.")


def process_shard(args, vocab_size):
    shard_id, shard = args
    tokenizer_model = get_tokenizer_model_path(vocab_size)
    enc = Tokenizer(tokenizer_model)
    with open(shard, "r") as f:
        data = json.load(f)
    all_tokens = []
    for example in tqdm(data, position=shard_id):
        text = example["story"]
        text = text.strip()  # get rid of leading/trailing whitespace
        tokens = enc.encode(text, bos=True, eos=False)  # encode the text, use BOS
        all_tokens.extend(tokens)
    # convert to uint16 nparray
    all_tokens = np.array(all_tokens, dtype=np.uint16)
    # calculate the output filename
    if vocab_size == 0:
        # if we're using Llama 2, just save the tokenized file in the same dir
        tokenized_filename = shard.replace(".json", ".bin")
    else:
        # save .bin files into a new tok{N} directory
        bin_dir = os.path.join(DATA_CACHE_DIR, f"tok{vocab_size}")
        shard_basename = os.path.basename(shard)
        bin_basename = shard_basename.replace(".json", ".bin")
        tokenized_filename = os.path.join(bin_dir, bin_basename)
    # write the bytes
    with open(tokenized_filename, "wb") as f:
        f.write(all_tokens.tobytes())
    # calculate the average sequence length (they are separated by BOS=1)
    avg_seq_len = all_tokens.size / ((all_tokens == 1).sum())
    print(f"Saved {tokenized_filename}, average seqlen: {avg_seq_len:.2f}")


def pretokenize(vocab_size):
    # iterate the shards and tokenize all of them one by one
    data_dir = os.path.join(DATA_CACHE_DIR, "TinyStories_all_data")
    shard_filenames = sorted(glob.glob(os.path.join(data_dir, "*.json")))
    if vocab_size > 0:
        # .bin files will be saved into tok{N} directory, create it once here
        bin_dir = os.path.join(DATA_CACHE_DIR, f"tok{vocab_size}")
        os.makedirs(bin_dir, exist_ok=True)

    # process all the shards in a process pool
    fun = partial(process_shard, vocab_size=vocab_size)
    with ProcessPoolExecutor() as executor:
        executor.map(fun, enumerate(shard_filenames))
    print("Done.")


class PretokDataset(torch.utils.data.IterableDataset):
    """
    Loads pretokenized examples from disk and yields them as PyTorch tensors.
    """

    def __init__(self, split, max_seq_len, vocab_size, vocab_source, data_cache_dir=None):
        super().__init__()
        self.split = split
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.vocab_source = vocab_source
        self.data_cache_dir = data_cache_dir or DATA_CACHE_DIR

    @staticmethod
    def _generate_rng_seed(worker_id, rank):
        """Generate a unique seed for random number generator."""
        return 42 + worker_id + 1337 * rank

    def _initialize_rng(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        rank = dist.get_rank() if dist.is_initialized() else 0
        seed = self._generate_rng_seed(worker_id, rank)
        return random.Random(seed)

    def _determine_bin_directory(self):
        """Determine the directory containing bin files based on the vocab source."""
        if self.vocab_source == "llama2":
            return os.path.join(self.data_cache_dir, "TinyStories_all_data")
        elif self.vocab_source == "custom":
            return os.path.join(self.data_cache_dir, f"tok{self.vocab_size}")
        else:
            raise ValueError(f"Unknown vocab source: {self.vocab_source}")

    def _get_shard_filenames(self):
        bin_dir = self._determine_bin_directory()
        return sorted(glob.glob(os.path.join(bin_dir, "*.bin")))

    def _filter_shard_filenames(self, shard_filenames):
        """Filter shard filenames based on train/test split."""
        return shard_filenames[1:] if self.split == "train" else shard_filenames[:1]

    def _read_shard_into_memory(self, shard):
        return np.memmap(shard, dtype=np.uint16, mode="r")

    def _create_batches_from_shard(self, shard_memmap):
        """Divide the shard data into batches based on the specified maximum sequence length."""
        num_batches = len(shard_memmap) // self.max_seq_len - 1
        if num_batches <= 0:
            raise ValueError("Shard too small. Investigate.")
        return list(range(num_batches))

    def __iter__(self):
        rng = self._initialize_rng()
        shard_filenames = self._get_shard_filenames()
        shard_filenames = self._filter_shard_filenames(shard_filenames)

        while True:
            rng.shuffle(shard_filenames)
            for shard in shard_filenames:
                shard_memmap = self._read_shard_into_memory(shard)
                batch_indices = self._create_batches_from_shard(shard_memmap)
                rng.shuffle(batch_indices)
                
                for ix in batch_indices:
                    start = ix * self.max_seq_len
                    end = start + self.max_seq_len + 1
                    chunk = torch.from_numpy(shard_memmap[start:end].astype(np.int64))
                    data,target = chunk[:-1], chunk[1:]
                    yield data,target 
# -----------------------------------------------------------------------------
# public interface functions

def get_tokenizer_model_path(vocab_size):
    """
    Returns path to the sentencepiece tokenizer model for a given vocab size
    vocab_size = 0 designates the default Llama 2 tokenizer, in that case
    None is returned.
    """
    if vocab_size == 0:
        return None
    else:
        return os.path.join(DATA_CACHE_DIR, f"tok{vocab_size}.model")

class Task:

    @staticmethod
    def iter_batches(batch_size, device, num_workers=0, **dataset_kwargs):
        ds = PretokDataset(**dataset_kwargs)
        dl = torch.utils.data.DataLoader(
            ds, batch_size=batch_size, pin_memory=True, num_workers=num_workers
        )
        for x, y in dl:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            yield x, y

# -----------------------------------------------------------------------------
# CLI for constructing the dataset

if __name__ == "__main__":
    vocab_source = "llama2" 
    device = torch.device("cpu")
    batch_size = 3
    max_seq_len =10
    vocab_size = 32000 
    split = "train"
    

    iter_batches = partial(
    Task.iter_batches,
    batch_size=batch_size,
    max_seq_len=max_seq_len,
    vocab_size=vocab_size,
    vocab_source=vocab_source,
    device=device,
    num_workers=0,
    )

    itr = batch_iter = iter_batches(split=split)

    for idx,(x,y) in enumerate(itr):
        print(idx)
    
    """
    These stages are designed to be run in order.

    To tokenize data with the Llama 2 tokenizer:
    python tinystories.py download
    python tinystories.py pretokenize

    To tokenize data with a custom tokenizer we train ourselves with sentencepiece, e.g.:
    python tinystories.py download
    python tinystories.py train_vocab --vocab_size=2048
    python tinystories.py pretokenize --vocab_size=2048
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", type=str, choices=["download", "pretokenize", "train_vocab"])
    parser.add_argument("--vocab_size", type=int, default=0, help="pretokenization vocab size. 0 = use Llama 2 tokenizer.")
    args = parser.parse_args()

    # depending on the stage call the appropriate function
    if args.stage == "download":
        download()
    elif args.stage == "train_vocab":
        train_vocab(vocab_size=args.vocab_size)
    elif args.stage == "pretokenize":
        pretokenize(vocab_size=args.vocab_size)
    else:
        raise ValueError(f"Unknown stage {args.stage}")
    

    