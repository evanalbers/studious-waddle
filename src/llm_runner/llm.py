import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class llm:
    """ a class representing the LLM and the running thereof """

    def __init__(self):
        """ initializes the LLM to GPT2 """
        self.model_name = "gpt2"
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name,
                                                           output_hidden_states=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def tokenize(self, text):
        """ tokenizes some given text using tokenizer 

        Params
        ------
        string : text
            some piece of text (a context) to be tokenized
        
        Returns
        -------
        tensors
            tensors representing tokenized text
        """

        return self.tokenizer(text, return_tensors="pt")

    def generate_activations(self, token):
        """ completes forward pass with given tokens

        Params
        ------
        tensor : token
            input tokens to be used in the pass
        
        Returns
        -------
        tensor
            activation vector of a given transformer in model - shape TBD
           """
        
        with torch.no_grad():
            outputs = self.model(**token)
            hidden_states = outputs.hidden_states

        return hidden_states
    

    def standard_forward_pass(self, token):
        """ completes a standard forward pass of the LLM
        Params
        ------
        tensor : token
            input tokens to be used in the pass

        Returns
        -------
        string : output
            output of the model
        """

        with torch.no_grad():
            outputs = self.model(**token)

        return outputs
    
    def hooked_forward_pass(self, token, hook_feature, layer):
        """ completes a "hooked" forward pass of LLM 
        Params
        ------
        token : tensor
            input tokens to be used in the pass

        hook_feature : tensor
            clamped activation values to be added to pass

        layer : int
            layer that values are being added to

        Returns
        -------
        string : outputs
            output of the model given "hooked" pass
        """
         
        hook = self.model.h.transformer.h[layer].register_forward_hook(hook_feature)

        with torch.no_grad():
            outputs = self.model(**token)

        hook.remove()

        return outputs




