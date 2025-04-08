from IterableCrawl import IterableCrawl
from llm import llm
from torch.utils.data import DataLoader
import torch
import os

def save_activations(activations, filename="activations.pt"):
    """ saves a given set of activations to the activations dataset file. """

    if os.path.exists(filename):
        data = torch.load(filename)
        
    else:
        data = torch.empty(activations.shape)

    print("activations shape:", activations.shape)
    print("file shale: ", data.shape)

    data = torch.cat((activations, data), dim=0)

    torch.save(data, filename)



dataset = IterableCrawl()

dataloader = DataLoader(dataset, batch_size = 4)
sample_llm = llm()

num_batches = 100

for num in range(num_batches):
    sample = next(iter(dataloader))
    tokens = sample_llm.tokenize(sample)
    activations = sample_llm.generate_activations(tokens, 5)
    print(len(activations))
    save_activations(activations)
    print("finished batch : ", num)







