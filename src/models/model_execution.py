from models.Gemini import *
from models.GPT import *
from models.Llama import *
import time
import pandas as pd

prompt = ''

models = ["Llama", "Gemini", "GPT 3.5", "GPT 4"]


def prompt_runner(model, prompt):
    # for model in models:
    # start_time = time.time()
    # print("Model: " + model)
    if model == "Llama":
        response = llama(prompt)
    if model == "Gemini":
        response = gemini(prompt)
    if model == "GPT 3.5":
        response = gpt3(prompt)
    if model == "GPT 4":
        response = gpt4(prompt)
    # print("Response: " + response)
    # print("Execution time: {:.2f} ms \n".format(
    #     (time.time()-start_time) * 1000))
    return response
