# config OPENAI

# !pip install --upgrade pip
# !pip install openai

import os
from openai import OpenAI
from constants.api import *


client = OpenAI(api_key=GPT_API_KEY)


def gpt_small(prompt):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return str(response.choices[0].message.content).strip()


def gpt_large(prompt):
    response = client.chat.completions.create(
        model="gpt-4-turbo", #
        messages=[{"role": "user", "content": prompt}]
    )
    return str(response.choices[0].message.content).strip()
