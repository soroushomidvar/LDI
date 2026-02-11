"""
OpenAI GPT model integration.

This module provides functions to interact with OpenAI's GPT models using the OpenAI API.
Supports both GPT-4o-mini (small, cost-effective) and GPT-4-turbo (large, high-performance) models.

Requirements:
    - openai package installed
    - OPENAI_API_KEY set in .env file at the project root
"""

import os
from openai import OpenAI

# Initialize OpenAI client with API key from environment
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def gpt_small(prompt: str) -> str:
    """
    Execute a prompt on OpenAI's GPT-4o-mini model.

    GPT-4o-mini is a smaller, faster, and more cost-effective version of GPT-4,
    suitable for tasks that don't require the full capabilities of the larger model.

    Args:
        prompt: The text prompt to send to the model

    Returns:
        The model's response as a string, with leading/trailing whitespace removed

    Raises:
        Exception: If the API call fails or returns an error
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return str(response.choices[0].message.content).strip()


def gpt_large(prompt: str) -> str:
    """
    Execute a prompt on OpenAI's GPT-4-turbo model.

    GPT-4-turbo is a high-performance model with advanced reasoning capabilities,
    suitable for complex tasks requiring sophisticated understanding and generation.

    Args:
        prompt: The text prompt to send to the model

    Returns:
        The model's response as a string, with leading/trailing whitespace removed

    Raises:
        Exception: If the API call fails or returns an error
    """
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return str(response.choices[0].message.content).strip()
