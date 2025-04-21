from data_generation import prepare_prompts_for_target 
from datasets import load_dataset
import torch
from rnn_generator import rnn
from train_rnn import generate_text


def test_target_prompt():
    """ testing that prepare_prompts_for_target displays appropriate behavior """

    ds = load_dataset("tiny_shakespeare", trust_remote_code=True)

    text = " ".join(ds['train']['text'])

    X, y, char_to_idx, idx_to_char, prompts = prepare_prompts_for_target(text, target_phrase="CORIOLANUS:", context_length=100) 

    print(f"Number of prompts: {len(prompts)}")
    print(f"Example prompt: {prompts[0]}")

def test_activation_shape():
    model_data = torch.load('shakespeare_rnn.pt', map_location=torch.device('cpu'))
    
    char_to_idx = model_data['char_to_idx']
    idx_to_char = model_data['idx_to_char']
    vocab_size = model_data['vocab_size']

    model = rnn(vocab_size)

    model.load_state_dict(model_data['model_state_dict'])



    model.eval()

    sample = generate_text(model,
                           seed_text="LEAR: ",
                           char_to_idx=char_to_idx,
                           idx_to_char=idx_to_char,
                           max_length=5000)
    
    print(sample)


    
if __name__ == "__main__":

    test_target_prompt()

