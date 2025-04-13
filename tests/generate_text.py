""" file that generates text prompts as text data """
import os
os.chdir("../")
from llm import llm
import torch


def generate_data(quantity, length, data_filename):
    """ generates some set of text regarding species of dogs and cats """

    data = torch.empty()

    sample_llm = llm()


    

