""" file to train an sae on an rnn network"""

import torch
from rnn_generator import rnn
from datasets import load_dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from autoencoder import SparseAutoEncoder
from tqdm import tqdm
from data_generation import prepare_shakespeare
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import time


def initialize_model(stream_width, hidden_width, lr, checkpoint_path=""):
    """ function that initializes the model from a checkpoint, or from scratch 
    Params
    ------
    stream_width : int
        width of stream (input/output size)

    hidden_width : int 
        width of hidden layer (should be > stream_width)

    lr : float
        learning rate for training
    
    checkpoint_path : string
        path to saved checkpoint

    Returns
    -------
    model : SparseAutoEncoder
        AutoEncoder to be trained
    
    optimizer : optim.Adam
        optimizer for the model
    
    start_epoch : int
        current epoch

    batch_num : int
        batch that training was paused at

    """
    print(checkpoint_path)
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model = SparseAutoEncoder(stream_width=stream_width, hidden_width=hidden_width, l1_penalty=0.0001)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer = optim.Adam(model.parameters(), lr=lr)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        batch_num = checkpoint.get('batch_num', 0)
        print(f"Resuming from epoch {start_epoch}, batch {batch_num}")

    else:
        model = SparseAutoEncoder(stream_width=stream_width, hidden_width=hidden_width, l1_penalty=0.0001)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        start_epoch = 0
        batch_num = 0

    return model, optimizer, start_epoch, batch_num


def generate_batch(dataloader_iter, llm, block):
    """ Returns a set of activations (equal to batch size)
    Params
    ------
    dataloader : DataLoader
        DataLoader to get iterator from (get next batch of data)

    llm : llm
        llm object used to generate tokens

    block : int
        int representing the layer block from which to retrieve activations

    Returns
    -------
    activations : tensor
        tensor representing the activations from the token pass in given block
    """

    sample = next(dataloader_iter)
    tokens = llm.tokenize(sample)
    activations = llm.generate_activations(tokens, block)

    return activations 

def save_checkpoint(checkpoint_path, epoch, batch_num, model_state_dict,
                     optimizer_state_dict, loss):
    """ saves checkpoint to a file to resume later training
    Params
    ------
    checkpoint_path : string
        path to save checkpoint to
    
    epoch : int
        epoch at the time of checkpoint saving

    batch_num : int
        batch number at time of saving
    
    model_state_dict : dict
        dictionary representing the model state at time of saving
    
    optimizier_state_dict : dict
        dictionary representing the optimizer state at time of saving
    
    loss : float 
        loss at time of saving

    Returns
    -------
    None
    
       """
    print(f"Time limit approaching. Saving checkpoint and exiting.")

    # save checkpoint to torch file
    torch.save({
        'epoch' : epoch,
        'batch_num' : batch_num,
        'model_state_dict' : model_state_dict,
        'optimizer_state_dict' : optimizer_state_dict,
        'loss' : loss
    }, checkpoint_path)


def train_sae(stream_width, hidden_width, epochs, lr, dataloader, layer, checkpoint_path, time_limit=3600):
    """ function that trains the sparse autoencoder
    Params
    ------
    stream_width : int
        width of stream (input/output size)

    hidden_width : int 
        width of hidden layer (should be > stream_width)

    epochs : int 
        number of epochs for training

    batches : int
        number of batches per epoch

    lr : float
        learning rate for training

    dataloader : DataLoader
        DataLoader to get iterator from (get next batch of data)

    llm : llm
        llm object used to generate tokens

    block : int
        int representing the layer block from which to retrieve activations

    Returns
    -------

    : bool
        represents whether the model has finished training or merely hit a checkpoint

    model : nn.model
        trained sparse autoencoder model

        """
    
    # clear out memory
    torch.cuda.empty_cache()

    # starting job timer 
    job_start_time = time.time()


    # init model
    model_data = torch.load('shakespeare_rnn.pt', map_location=torch.device('cpu'))
    
    char_to_idx = model_data['char_to_idx']
    idx_to_char = model_data['idx_to_char']
    vocab_size = model_data['vocab_size']

    lang_model = rnn(vocab_size)

    lang_model.load_state_dict(model_data['model_state_dict'])
    lang_model.eval()

    sae_model = SparseAutoEncoder(stream_width=stream_width, hidden_width=hidden_width, l1_penalty=0.0001)

    optimizer = optimizer = optim.Adam(sae_model.parameters(), lr=lr)

    criterion = nn.MSELoss()
    
    # main training loop
    for epoch in range(0, epochs):

        # clear out memory
        torch.cuda.empty_cache()    
        
        # progress bar
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=True)
        epoch_loss = 0
        count = 0
        for x_batch, y_batch in pbar:

            with torch.no_grad():
                _, _, activations = lang_model.forward(x_batch, return_activations=True)
                activations = activations[layer].detach()
        
            # train model on an additional batch of activations
            
            optimizer.zero_grad()
            reconstruction = sae_model(activations)

            # include l1 penalty to ensure sparseness
            loss = criterion(reconstruction, activations) + sae_model.get_l1_loss(activations)
            loss.backward()
            optimizer.step()

            # memory optimization 
            
            del reconstruction
            torch.cuda.empty_cache()

            # update statistics
            epoch_loss += loss.item()
            count += 1
            current_epoch_avg_loss = epoch_loss / count
            
            # updating progress bar
            pbar.set_postfix({"avg_loss": f"{current_epoch_avg_loss:.4f}", "batch_loss": f"{loss.item():.4f}"})

        # epoch summary
        # print(f"Epoch {epoch + 1}/{epochs} completed, average loss: {epoch_loss/batches:.4f}")

        # save a checkpoint post epoch, just in case
        save_checkpoint(checkpoint_path, epoch + 1, 0, sae_model.state_dict(), 
                        optimizer.state_dict(), epoch_loss)


    return True, sae_model

if __name__ == "__main__":

    ds = load_dataset("tiny_shakespeare", trust_remote_code=True)

    dataloader, char_to_idx, idx_to_char, vocab_size = prepare_shakespeare(ds["train"])

    stream = 512

    hidden = 2000

    epochs = 2

    block = 0

    sae = train_sae(stream, hidden, epochs, lr=0.001, 
                    dataloader=dataloader, layer=block, checkpoint_path="layer1_sae.pt", time_limit=120)