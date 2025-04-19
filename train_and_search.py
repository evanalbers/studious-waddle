""" script that makes a series of autoencoders and trains an autoencoder on each of their layers """

from train_sae import train_sae
from llm import llm
from torch.utils.data import DataLoader
from generate_text import generate_prompts
from test_feature_visualization import load_model, generate_activations, calculate_log_likelihoods, plot_single_feature

def train_mini_sae(block):
    """ trains mini sae on given layer"""

    prompt_set = generate_prompts(100000, 512)
    
    dataloader = DataLoader(prompt_set, batch_size=4)

    sample_llm = llm()

    stream = 768

    hidden = 1000

    epochs = 1

    batches = 100

    sae = train_sae(stream, hidden, epochs, batches, 0.001, 
                    dataloader=dataloader, llm=sample_llm, block=block, checkpoint_path="large_sae/layer" + str(block) + ".pt", time_limit=1200)

    return sae

if __name__ == "__main__":

    for layer in range(13):
        train_mini_sae(layer)

    for layer in range(1, 13):
        sample_llm = llm()

        model = load_model("large_sae/layer" 
                           + str(layer) + ".pt")

        tokens, animal, breed, activations = generate_activations(model, sample_llm, 500, block=layer)

        

        for ani in range(2):
            for breed_type in range(10):
                print(activations)
                plot_single_feature(activations, 5, categories=breed, animal=animal, target_animal=ani, target_category=breed_type, block=layer)

