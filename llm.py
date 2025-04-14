import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class llm:
    """ a class representing the LLM and the running thereof """

    def __init__(self):
        """ initializes the LLM to GPT2 """

        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU device count: {torch.cuda.device_count()}")
        if torch.cuda.is_available():
            print(f"Current GPU: {torch.cuda.get_device_name()}")

        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Set default precision based on device
        if self.device.type == "cuda":
            self.dtype = torch.float16  # Use half precision on GPU by default
        else:
            self.dtype = torch.float32
            
        print(f"Using device: {self.device}, dtype: {self.dtype}")

        self.model_name = "gpt2"
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name,
                                                           output_hidden_states=True,
                                                           return_dict_in_generate=True,
                                                           torch_dtype=self.dtype,
                                                           device_map=self.device,
                                                           use_cache=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token


    def __str__(self):
        """ string object of underlying model
        Params
        ------
        None

        Returns
        -------
        name : string 
        """

        return str(self.model)
    

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

        return self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=256)

    def generate_activations(self, token, layer, return_output=False):
        """ completes forward pass with given tokens

        Params
        ------
        tensor : token
            input tokens to be used in the pass

        int : layer
            block whose activations we wish to see. Must be > 1.
        
        Returns
        -------
        tensor
            activation vector of a given transformer in model - shape TBD
           """
        
        inputs = {k: v.to(self.device) for k, v in token.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            hidden_states = outputs.hidden_states[layer]

        if return_output:
            return outputs, hidden_states

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




