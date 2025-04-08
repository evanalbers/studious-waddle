import torch

from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset

class IterableCrawl(IterableDataset):
    """ A class that generates a pipeline of  prompts for training the autoencoder """

    def __init__(self):
        """ sets up pipeline """
        super().__init__()
        self.dataset = load_dataset("allenai/c4", "en", split="train", streaming=True, trust_remote_code=True)

    def __iter__(self):
        """ returns yield object for iterating over samples in the dataset """
        for sample in self.dataset:
            yield sample["text"]

