""" job script to train all three saes on shakespeare rnn """
from datasets import load_dataset
from data_generation import prepare_shakespeare
from train_rnn_sae import train_sae
if __name__ == "__main__":

    ds = load_dataset("tiny_shakespeare", trust_remote_code=True)

    dataloader, char_to_idx, idx_to_char, vocab_size = prepare_shakespeare(ds["train"])

    stream = 512

    hidden = 2000

    epochs = 4

    block = 0

    sae = train_sae(stream, hidden, epochs, lr=0.001, 
                    dataloader=dataloader, layer=block, checkpoint_path="layer1_sae.pt", time_limit=120)
    
    block = 1

    sae = train_sae(stream, hidden, epochs, lr=0.001, 
                    dataloader=dataloader, layer=block, checkpoint_path="layer2_sae.pt", time_limit=120)
    
    block = 2

    sae = train_sae(stream, hidden, epochs, lr=0.001, 
                    dataloader=dataloader, layer=block, checkpoint_path="layer3_sae.pt", time_limit=120)
    
    