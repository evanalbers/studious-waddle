""" file to visualize the frequency of mentioning coriolanus when next few characters should be coriolanus """
import matplotlib.pyplot as plt
from data_generation import prepare_prompts_for_target
from train_rnn import generate_text
import numpy as np
import torch
from rnn_generator import rnn
from datasets import load_dataset
from tqdm import tqdm

def generate_frequency_data(model, text, target, temp, max_context=100):

    """ generates frequency data for a given prompt, temp """
    y = []

    pbar = tqdm(range(1, max_context+1), desc="Training on contexts of max lengths")

    for context_length in pbar:
        _, _, char_to_idx, idx_to_char, prompts = prepare_prompts_for_target(text, target, context_length=context_length)

        positive_samples = []
        for prompt in prompts:
            sample = generate_text(model, prompt, char_to_idx=char_to_idx, idx_to_char=idx_to_char, max_length=100, temperature=temp)
            if target in sample:
                positive_samples.append(sample)


        # print(len(positive_samples))
        positive_frequency = len(positive_samples) / len(prompts)
        y.append(positive_frequency)

    return y



def generate_frequency_plots(model, text, prompt, max_context):
    """ generates a plot of frequency of mentioning versus character context for multiple temperatures """

    y_data = []

    temperatures = np.linspace(0.1, 1, 5)

    print(temperatures)

    for temp in temperatures:
        y = generate_frequency_data(model, text, prompt, temp, max_context=max_context)

        y_data.append(y)

    x = np.linspace(1, max_context, max_context)



    fig, ax = plt.subplots(figsize=(10, 6))
    for temp in range(temperatures.size):
        ax.plot(x, y_data[temp], label=f'Temperature={temperatures[temp]}')

    ax.legend()
    ax.set_xlabel("Prompt Context Length")
    ax.set_ylabel("Frequency of Target Phrase Occurences in Gen. Text")
    ax.set_title("Target Phrase Frequency Vs. Prompt Length, Multiple Temps. ")

    plt.show()

if __name__ == '__main__':

    model_data = torch.load('shakespeare_rnn.pt', map_location=torch.device('cpu'))
    
    char_to_idx = model_data['char_to_idx']
    idx_to_char = model_data['idx_to_char']
    vocab_size = model_data['vocab_size']

    model = rnn(vocab_size)

    model.load_state_dict(model_data['model_state_dict'])
    model.eval()

    ds = load_dataset("tiny_shakespeare", trust_remote_code=True)

    text = " ".join(ds['train']['text'])

    generate_frequency_plots(model, text, "CORIOLANUS:", 50)

