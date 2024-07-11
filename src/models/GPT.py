# config OPENAI

# !pip install --upgrade pip
# !pip install openai

import os
from openai import OpenAI
from constants.api import *


client = OpenAI(api_key=GPT_API_KEY)


def gpt3(prompt):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return str(response.choices[0].message.content).strip()


def gpt4(prompt):
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return str(response.choices[0].message.content).strip()
