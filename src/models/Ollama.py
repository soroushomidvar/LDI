from ollama import chat
from ollama import ChatResponse

def ollama(model, prompt):

    response: ChatResponse = chat(model=model, messages=[{'role': 'user','content': prompt,},])
    r = str(response.message.content)
    
    return r
