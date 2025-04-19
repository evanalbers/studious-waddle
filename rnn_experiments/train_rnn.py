""" file containing functions to train rnn. """
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset
from rnn_generator import rnn



def train_rnn(model, data_loader, epochs, learning_rate=0.0001, clip_grad=0.5):
    """ trains rnn """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        
        model.train()
        total_loss = 0

        pbar = tqdm(data_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=True)

        for x_batch, y_batch in pbar:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            model.reset_hidden()
            output = model(x_batch)

            output = output[:, -1, :]

            loss = criterion(output, y_batch)

            optimizer.zero_grad()
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

            optimizer.step()

            total_loss += loss.item()

            pbar.set_postfix(loss=f'{loss.item():.4f}')
         

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




def generate_text(model, seed_text, char_to_idx, idx_to_char, max_length=1000, temperature=.7):
    """ gen text using trained model """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    chars = [char_to_idx[ch] for ch in seed_text]
    input_tensor = torch.LongTensor([chars]).to(device)

    generated_text = seed_text

    model.reset_hidden()

    with torch.no_grad():
        for _ in range(max_length):

            output = model(input_tensor)

            output = output[:, -1, :] / temperature
            probs = torch.softmax(output, dim=1)
            
            next_char_idx = torch.multinomial(probs, 1).item()

            generated_text += idx_to_char[next_char_idx]

            input_tensor = torch.LongTensor([[next_char_idx]]).to(device)

    return generated_text

if __name__ == "__main__":

    ds = load_dataset("tiny_shakespeare", trust_remote_code=True)

    dataloader, char_to_idx, idx_to_char, vocab_size = prepare_shakespeare(ds["train"])

    model = rnn(vocab_size)

    train_rnn(model, dataloader, 1)

    sample = generate_text(model,
                           seed_text="ROMEO: ",
                           char_to_idx=char_to_idx,
                           idx_to_char=idx_to_char,
                           max_length=200)
    
    print("\nGenerated Sample:")
    print(sample)

    torch.save({
        'model_state_dict': model.state_dict(),
        'char_to_idx' : char_to_idx,
        'idx_to_char' : idx_to_char,
        'vocab_size' : vocab_size

    }, 'shakespeare_rnn.pt')

    # model_data = torch.load('shakespeare_rnn.pt')
    
    # char_to_idx = model_data['char_to_idx']
    # idx_to_char = model_data['idx_to_char']
    # vocab_size = model_data['vocab_size']

    # model = rnn(vocab_size)

    # model.load_state_dict(model_data['model_state_dict'])



    # model.eval()

    # sample = generate_text(model,
    #                        seed_text="LEAR: ",
    #                        char_to_idx=char_to_idx,
    #                        idx_to_char=idx_to_char,
    #                        max_length=5000)
    
    # print(sample)






