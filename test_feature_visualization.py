""" testing feature visualization on cat dog dataset 
Expecting somewhere around 22 or so distinct features? Two types of possible stories, ten breeds per.
"""
import torch
from tqdm import tqdm 
from generate_text import generate_prompt

from autoencoder import SparseAutoEncoder

def load_model(filepath):
    """ loads model, returns given autoencoder """

    checkpoint = torch.load("test_checkpoint.pt")

    model = SparseAutoEncoder(768, 1000, 0.0001)

    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()

    return model


# need to collect data from autoencoder 

def generate_activations(model, llm, num_activations):
    """ gets activations from prompts that are generated """

    activations = []
    breed = []
    animal = []
    tokens = []

    # data_size = len(data.dataset)

    pbar = tqdm(range(num_activations), initial=0,
                     total=num_activations, desc=f"Generating activations", leave=True)
    
    for iteration_idx in pbar:

        dog_prompt, dog, dog_breed = generate_prompt(1, 512)
        cat_prompt, cat, cat_breed = generate_prompt(0, 512)

        dog_tokens = llm.tokenize(dog_prompt)
        cat_tokens = llm.tokenize(cat_prompt)
        
        dog_activations = llm.generate_activations(dog_tokens)
        dog_feature_activations = model.get_activations(dog_activations)

        cat_activations = llm.generate_activations(cat_tokens)
        cat_feature_activations = model.get_activations(cat_activations)

        tokens.append(dog_tokens)
        tokens.append(cat_tokens)

        animal.append(dog)
        animal.append(cat)

        breed.append(dog_breed)
        breed.append(cat_breed)

        activations.append(dog_feature_activations)
        activations.append(cat_feature_activations)

        return tokens, animal, breed, activations
    

        

        





# then calculate log-likelihoods 

# 