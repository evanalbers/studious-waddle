""" file representing code used to train the malicious prompt-generating model """
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained('gpt2')
tokenizer = AutoTokenizer.from_pretrained('gpt2')
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

def preference_function(prompt, response_a, response_b):
    """ function that determines which response is preferred from the evil LLM 
    Params
    ------
    prompt : string
        prompt for malicious LLM - something like "give me a prompt to get an LLM to do X"

    response_a : string
        one of two responses from malicious LLM (prompt for regular LLM)

    response_b : string
        second of two responses from malicious LLM (prompt for regular LLM)
    
    Returns
    -------
    preference_score : int
        0 or 1, representing which prompt is "preferred" - generated a higher activation in the SAE. 
        1 if response_a, 0 if response_b
    """

    return preference_score 

def process_single_prompt_set(prompt, response_a, response_b):
    """ optimizes the prompt model on a single instance of comparison prompts 
    Params
    ------
    prompt : string
        prompt for malicious LLM - something like "give me a prompt to get an LLM to do X"

    response_a : string
        one of two responses from malicious LLM (prompt for regular LLM)

    response_b : string
        second of two responses from malicious LLM (prompt for regular LLM)
    
    """

    pref = preference_function(prompt, response_a, response_b)

    tokens_a = tokenizer(prompt + response_a, return_tensors='pt')

    tokens_b = tokenizer(prompt + response_b, return_tensors='pt')

    log_p_a = -model(**tokens_a, labels=tokens_a.input_ids).loss
    log_p_b = -model(**tokens_b, labels=tokens_b.input_ids).loss

    beta = 0.1

    # implementing DPO Loss
    if pref == 1:
        loss = -torch.log(torch.sigmoid(beta * (log_p_a - log_p_b)))

    else:
        loss = -torch.log(torch.sigmoid(beta * (log_p_b - log_p_a)))

    # updating the model
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()
