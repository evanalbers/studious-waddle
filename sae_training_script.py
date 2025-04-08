""" script file to train sparse autoencoder on Discovery Cluster """
from IterableCrawl import IterableCrawl
from torch.utils.data import DataLoader
from llm import llm
from train_sae import train_sae
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, default="checkpoint.pt")
    parser.add_argument("--time_limit", type=int, default=3600)
    args = parser.parse_args()


    dataset = IterableCrawl()

    dataloader = DataLoader(dataset, batch_size = 4)
    sample_llm = llm()

    stream = 768

    hidden = 10 * stream

    epochs = 2

    batches = 500000

    block = 5

    training_complete, sae = train_sae(stream,
                    hidden,
                    epochs,
                    batches,
                    0.001, 
                    dataloader=dataloader,
                    llm=sample_llm,
                    block=block,
                    checkpoint_path=args.checkpoint_path,
                    time_limit=args.time_limit)

    if training_complete:
        print("Training complete!")

    else: 
        print("Checkpoint reached!")