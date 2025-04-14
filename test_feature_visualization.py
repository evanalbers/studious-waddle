""" testing feature visualization on cat dog dataset 
Expecting somewhere around 22 or so distinct features? Two types of possible stories, ten breeds per.
"""
import torch
from tqdm import tqdm 
from generate_text import generate_prompt
from autoencoder import SparseAutoEncoder
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from llm import llm


def load_model(filepath):
    """ loads model, returns given autoencoder """

    checkpoint = torch.load(filepath)

    model = SparseAutoEncoder(768, 22, 0.0001)

    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()

    return model


# need to collect data from autoencoder 

def generate_activations(model, llm, num_activations, block):
    """ gets activations from prompts that are generated """

    all_activations = []
    breed_info = []
    animal_info = []
    all_tokens = []

    pbar = tqdm(range(num_activations), initial=0,
                     total=num_activations, desc=f"Generating activations", leave=True)
    
    for iteration_idx in pbar:

        dog_prompt, dog, dog_breed = generate_prompt(1, 512)
        cat_prompt, cat, cat_breed = generate_prompt(0, 512)

        dog_tokens = llm.tokenize(dog_prompt)
        cat_tokens = llm.tokenize(cat_prompt)
        
        dog_activations = llm.generate_activations(dog_tokens, block)
        dog_feature_activations = model.get_activations(dog_activations)

        cat_activations = llm.generate_activations(cat_tokens, block)
        cat_feature_activations = model.get_activations(cat_activations)

        all_tokens.append(dog_tokens)
        all_tokens.append(cat_tokens)
        
        # Flatten activations and append corresponding metadata
        for i in range(dog_feature_activations.shape[1]):
            # print(dog_feature_activations[0][i].detach().numpy().shape)
            all_activations.append(dog_feature_activations[0][i].detach().numpy())
            animal_info.append(dog)
            breed_info.append(dog_breed)
            
        for i in range(cat_feature_activations.shape[1]):
            all_activations.append(cat_feature_activations[0][i].detach().numpy())
            animal_info.append(cat)
            breed_info.append(cat_breed)
    
    return all_tokens, np.array(animal_info), np.array(breed_info), np.stack(all_activations)
    
def calculate_log_likelihoods(activations, categories, target_category):
    """ Calculate log likelihood ratios for breed detection using an approach 
    similar to the Arabic script detection method."""
    
    # Calculate P(breed) - prior probability of the target breed
    # Assuming each breed appears with equal frequency (1/10)
    p_target_breed = 1/10
    p_other_breeds = 9/10
    
    # Create bins for histograms
    bins = np.linspace(0, np.max(activations), 100)
    
    # Create a list to store results with consistent shape
    log_likelihood_ratios = []
    
    for feature_idx in range(activations.shape[1]):
        feature_values = activations[:, feature_idx]
        
        # Calculate P(feature|breed) using histograms
        target_indices = (categories == target_category)
        
        # Use similar threshold approach as in Arabic example
        # If activation is above a threshold, consider it "strongly indicative"
        mean_val = np.mean(feature_values)
        std_val = np.std(feature_values)
        threshold = mean_val + std_val
        
        # For each bin, calculate the log-likelihood ratio
        bin_ratios = []
        for i in range(len(bins)-1):
            # Count samples in this bin
            bin_start, bin_end = bins[i], bins[i+1]
            bin_indices = (feature_values >= bin_start) & (feature_values < bin_end)
            
            # Count how many samples in this bin are from target breed
            target_in_bin = np.sum(bin_indices & target_indices)
            total_in_bin = np.sum(bin_indices)
            
            # Similar to how Arabic uses a strong prior for full Arabic text
            if total_in_bin > 0:
                # P(breed|feature_in_bin) 
                p_bin_given_target = target_in_bin / np.sum(target_indices)
                p_bin_given_other = (total_in_bin - target_in_bin) / np.sum(~target_indices)
                
                # Avoid division by zero
                p_bin_given_other = max(p_bin_given_other, 1e-10)
                p_bin_given_target = max(p_bin_given_target, 1e-10)
                
                # Use likelihood ratio: P(bin|target) / P(bin|other)
                ratio = p_bin_given_target / p_bin_given_other
                
                # Adjust with prior
                ratio = ratio * (p_target_breed / p_other_breeds)
                
                # Calculate log
                log_ratio = np.log(ratio)
            else:
                # If no samples in this bin, use a neutral value
                log_ratio = 0.0
                
            bin_ratios.append(log_ratio)
        
        # Store feature index and a single scalar value (mean of bin ratios)
        # This ensures all elements have the same shape
        log_likelihood_ratios.append((feature_idx, np.mean(bin_ratios)))
    
    return log_likelihood_ratios, bins

def plot_single_feature(activations, feature, categories, animal, target_category=5, target_animal = 0):
    """ plotting activation of a single feature """

    # Create figure
    # plt.figure()

    target_indices = (categories == target_category) & (animal == target_animal)  # needs to be modded so that it reflects cats and dogs, this is double what it ought to be

    best = 1

    max = 0

    print(activations[target_indices].shape)

    for num in range(activations.shape[1]):

        if np.sum(activations[target_indices, num]) > max:
            best = num
            max = np.sum(activations[target_indices])
            print(best)

    print(len(categories))
 
    # getting activations of a specific feature
    feature_activations = activations[:, best]

    print(feature_activations.shape)
    print(target_indices)

    sns.displot(x=feature_activations, kind='kde', hue=target_indices, fill=True, multiple='stack')

    plt.xlabel('Activation Level')
    plt.ylabel('Density')

    plt.show()




if __name__ == "__main__":

    sample_llm = llm()

    model = load_model("test_checkpoint.pt")

    tokens, animal, breed, activations = generate_activations(model, sample_llm, 1000, block=5)

    llr, bins = calculate_log_likelihoods(activations, breed, target_category=5)

    for ani in range(2):
        for breed_type in range(10):
            plot_single_feature(activations, 5, categories=breed, animal=animal, target_animal=ani, target_category=breed_type)

    # plot_top_discriminative_features(activations, breed, target_category=5, llr_values=llr)

    # plot_feature_activation_distribution(activations, breed, tokens, llr, bins, target_category=5)


