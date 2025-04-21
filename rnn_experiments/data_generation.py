""" file that contains functions to generate datasets for rnn """
import torch 
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset

def prepare_data(text, seq_length=50, step=1):
    """ prepares char sequences from text for training """

    # creating char to index mapping
    chars = sorted(list(set(text)))
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}

    seq = []
    next_chars = []

    for i in range(0, len(text) - seq_length, step):
        seq.append(text[i:i + seq_length])
        next_chars.append(text[i + seq_length])

    X = torch.zeros((len(seq), seq_length), dtype=torch.long)
    y = torch.zeros(len(seq), dtype=torch.long)

    for i, (sequence, next_char) in enumerate(tqdm(zip(seq, next_chars), total=len(seq), desc="Converting to tensors")):
        for t, char in enumerate(sequence):
            X[i, t] = char_to_idx[char]
        y[i] = char_to_idx[next_char]

    return X, y, char_to_idx, idx_to_char

def prepare_shakespeare(data, seq_length=100, batch_size=64):
    """ processes tiny shakespeare dataset for training """

    text = " ".join(data['text'])
    
    X, y, char_to_idx, idx_to_char = prepare_data(text, seq_length)

    dataset = TensorDataset(X, y)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader, char_to_idx, idx_to_char, len(char_to_idx)

def prepare_prompts_for_target(text, target_phrase, context_length=100, step=1, verbose=False):
    """ generates prompts based on given target phrase """

    chars = sorted(list(set(text)))
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}
    
    prompts = []
    target_indices = []
    
    # track target phrase by starting index 
    start_idx = 0
    while True:
        idx = text.find(target_phrase, start_idx)
        if idx == -1:
            break
        target_indices.append(idx)
        start_idx = idx + 1
    
    if verbose:
        print(f"Found {len(target_indices)} occurrences of '{target_phrase}'")
    
    # for each index of starting phrase
    for idx in target_indices:
        if idx >= context_length:
            
            # If we have enough preceding context
            prompts.append(text[idx - context_length:idx])
        else:
            
            # If we don't have enough context, pad with spaces
            prompts.append(" " * (context_length - idx) + text[:idx])
    
    # Convert to tensors
    X = torch.zeros((len(prompts), context_length), dtype=torch.long)
    y = torch.zeros(len(prompts), dtype=torch.long)
    
    # Fill tensors
    for i, prompt in enumerate(tqdm(prompts, desc="Converting to tensors",  disable=not verbose)):
        for t, char in enumerate(prompt):
            X[i, t] = char_to_idx[char]
        # Set target to the first character of the target phrase
        y[i] = char_to_idx[target_phrase[0]]
    
    return X, y, char_to_idx, idx_to_char, prompts
