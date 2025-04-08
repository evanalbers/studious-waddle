""" Training script for sparse autoencoder """
import torch
from IterableCrawl import IterableCrawl
from llm import llm
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from autoencoder import SparseAutoEncoder
from tqdm import tqdm
import os
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
        model = SparseAutoEncoder(stream_width=stream_width, hidden_width=hidden_width)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer = optim.Adam(model.parameters(), lr=lr)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        batch_num = checkpoint.get('batch_num', 0)
        print(f"Resuming from epoch {start_epoch}, batch {batch_num}")

    else:
        model = SparseAutoEncoder(stream_width=stream_width, hidden_width=hidden_width)
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


def train_sae(stream_width, hidden_width, epochs, batches, lr, dataloader, llm, block, checkpoint_path, time_limit=3600):
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
    
    # starting job timer 
    job_start_time = time.time()

    # initializing model
    model, optimizer, start_epoch, batch_num = initialize_model(stream_width, hidden_width, lr, checkpoint_path)

    criterion = nn.MSELoss()
    
    # main training loop
    for epoch in range(start_epoch, epochs):

        dataloader_iter = iter(dataloader)

        # updating iterator if necessary
        if epoch == start_epoch and batch_num > 0:
            for _ in range(batch_num):
                next(dataloader_iter, None)

        remaining_batches = batches - batch_num

        # progress bar
        pbar = tqdm(range(remaining_batches), initial=batch_num,
                     total=batches, desc=f"Epoch {epoch+1}/{epochs}", leave=True)
        epoch_loss = 0

        for batch_idx in pbar:

            # check time remaining
            current_time = time.time()
            if current_time - job_start_time > (time_limit - 30):

                # calculate loss and save 
                loss = epoch_loss / (batch_idx + 1) if batch_idx > 0 else 0
                save_checkpoint(checkpoint_path, epoch, batch_idx,
                                 model.state_dict(), optimizer.state_dict(), loss)
                
                print(f"Paused training on batch {batch_idx} out of {batches}")
                print(f"Training time: {current_time - job_start_time:.2f} seconds")
                print(f"Checkpoint saved. Resume to continue training.")
                return False, model

            # train model on an additional batch of activations
            batch = generate_batch(dataloader_iter=dataloader_iter, llm=llm, block=block)
            optimizer.zero_grad()
            reconstruction = model(batch)
            loss = criterion(reconstruction, batch)
            loss.backward()
            optimizer.step()

            # update statistics
            epoch_loss += loss.item()
            current_epoch_avg_loss = epoch_loss / (batch_idx + 1)
            
            # updating progress bar
            pbar.set_postfix({"avg_loss": f"{current_epoch_avg_loss:.4f}", "batch_loss": f"{loss.item():.4f}"})

        # epoch summary
        print(f"Epoch {epoch + 1}/{epochs} completed, average loss: {epoch_loss/batches:.4f}")

        # save a checkpoint post epoch, just in case
        save_checkpoint(checkpoint_path, epoch + 1, 0, model.state_dict(), 
                        optimizer.state_dict(), epoch_loss/remaining_batches)


    return True, model


if __name__ == "__main__":

    dataset = IterableCrawl()

    dataloader = DataLoader(dataset, batch_size = 4)
    sample_llm = llm()

    stream = 768

    hidden = 10 * stream

    epochs = 2

    batches = 500000

    block = 5

    sae = train_sae(stream, hidden, epochs, batches, 0.001, 
                    dataloader=dataloader, llm=sample_llm, block=block, checkpoint_path="testing_checkpoint.pt", time_limit=120)