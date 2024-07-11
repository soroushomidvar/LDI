# config Llama

# !pip install replicate

import replicate
import os
from constants.api import *

os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_KEY


def llama(msg):
    output = replicate.run("meta/llama-2-70b-chat", input={"prompt": msg})
    r = ""
    for item in output:
        r += item

    return r.replace("\n", " ")
