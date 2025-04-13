""" file that generates text prompts as text data """
import os
# os.chdir("./")
print(os.getcwd())
from llm import llm
import torch
import numpy as np
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence


def generate_data(quantity, length, response_data_filename, activation_data_filename):
    """ generates some set of text regarding species of dogs and cats """

    sample_llm = llm()

    prompt = "Tell me a short story, of around " + str(length) + " tokens, about a "

    dog_breeds = ["German Shepherd", "Bulldog",
                  "Labrador Retriever", "French Bulldog",
                  "Siberian Husky", "Beagle",
                  "Poodle", "Chihuahua",
                  "Dachshund", "dog"]
    
    cat_breeds = ["Siamese cat", "British Shorthair cat", 
                  "Maine Coon cat", "Persian cat", 
                  "Sphynx cat", "Abyssinian cat",
                  "Burmese cat", 'Scottish Fold cat', 
                  "Himalayan cat", "cat"]

    rand = np.random.default_rng()

    pbar = tqdm(range(quantity), initial = 0)

    for iteration_idx in pbar:
        
        dog_index = rand.integers(low=0, high=9)
        cat_index = rand.integers(low=0, high=9)

        dog_prompt = prompt + dog_breeds[dog_index]
        cat_prompt = prompt + cat_breeds[cat_index]

        dog_tokens = sample_llm.tokenize(dog_prompt, )
        cat_tokens = sample_llm.tokenize(cat_prompt, )

        dog_response, dog_activations = sample_llm.generate_activations(dog_tokens, 5, return_output=True)
        cat_response, cat_activations = sample_llm.generate_activations(cat_tokens, 5, return_output=True)
        if iteration_idx == 0:
            data = {
                'label': [],
                'breed_idx': [],
                'activations': [],
                'response': []
            }


        # adding dog ex
        data['label'].append(1)
        data['breed_idx'].append(dog_index)
        data['activations'].append(dog_activations.squeeze(0))
        data['response'].append(dog_response)

        # adding cat ex
        data['label'].append(1)
        data['breed_idx'].append(cat_index)
        data['activations'].append(cat_activations.squeeze(0))
        data['response'].append(cat_response)

    response_data = {

        'label' : torch.tensor(data['label']),
        'breed_idx' : torch.tensor(data['breed_idx']),
        'response' : data['response']

    }

    activation_data = torch.cat(data['activations'])

    torch.save(response_data, response_data_filename)
    torch.save(activation_data, activation_data_filename)



if __name__ == "__main__":

    generate_data(1000, 500, "test_data_responses.pt", "test_data_activations.pt")





    

