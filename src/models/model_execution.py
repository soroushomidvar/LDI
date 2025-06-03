#from models.Gemini import *
# from models.Llama import *
from models.GPT import *
from models.Ollama import *
import time
import pandas as pd

prompt = ''

models = ["Llama Small", "Gemini", "GPT Small", "GPT Large"]


def prompt_runner(model, prompt):
    # for model in models:
    # start_time = time.time()
    # print("Model: " + model)
    if model == "Llama Small":
        response = ollama("llama3.2:3b", prompt)
    # if model == "Gemini":
    #     response = gemini(prompt)
    if model == "GPT Small":
        response = gpt_small(prompt)
    if model == "GPT Large":
        response = gpt_large(prompt)
    # print("Response: " + response)
    # print("Execution time: {:.2f} ms \n".format(
    #     (time.time()-start_time) * 1000))
    return response
