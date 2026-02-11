from models.GPT import gpt_small, gpt_large
from models.Ollama import ollama

# Available model options for prompt execution
MODELS = ["Llama Small", "Gemini", "GPT Small", "GPT Large"]


def prompt_runner(model: str, prompt: str) -> str:
    """
    Execute a prompt on a specified language model and return the response.

    Args:
        model: Name of the model to use. Supported values:
            - "Llama Small": Uses Ollama with llama3.2:3b model
            - "GPT Small": Uses OpenAI GPT-4o-mini
            - "GPT Large": Uses OpenAI GPT-4-turbo
        prompt: The text prompt to send to the model

    Returns:
        The model's response as a string

    Note:
        Gemini model support is currently commented out and not available.
    """
    if model == "Llama Small":
        response = ollama("llama3.2:3b", prompt)
    elif model == "GPT Small":
        response = gpt_small(prompt)
    elif model == "GPT Large":
        response = gpt_large(prompt)
    else:
        raise ValueError(
            f"Unsupported model: {model}. Available models: {MODELS}")

    return response
