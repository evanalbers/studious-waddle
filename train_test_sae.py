""" file script to train test SAE """
import torch
from generate_text import generate_prompts
from torch.utils.data import DataLoader
from train_sae import train_sae
from llm import llm


if __name__ == "__main__":

    prompt_set = generate_prompts(100000, 512)
    
    dataloader = DataLoader(prompt_set, batch_size=4)

    sample_llm = llm()

    stream = 768

    hidden = 1000

    epochs = 2

    batches = 2500

    block = 5

    sae = train_sae(stream, hidden, epochs, batches, 0.001, 
                    dataloader=dataloader, llm=sample_llm, block=block, checkpoint_path="test_small_checkpoint.pt", time_limit=600)

    