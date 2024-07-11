from models.model_execution import *
from constants.prompts import *
import spacy
import json
import re


def text_annotator(text, method="spacy"):

    r = None

    def LLMs():
        fixed_initial = ALL_MODELS_NER_PROMPTS.VALUE['NER_FIXED_INITIAL']
        fixed_querry = ALL_MODELS_NER_PROMPTS.VALUE['NER_FIXED_QUERRY']
        prompt = fixed_initial + text + fixed_querry
        print(prompt)
        response = prompt_runner(method, prompt)
        print(response)
        # remove whitespace
        return str(response)

    def spaCy():
        annotated_text = ""
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)
        # Annotate named entities
        for ent in doc.ents:
            annotated_text += f"{ent.text} [{ent.label_}] "
        return annotated_text.strip()

    if method == "Llama" or method == "GPT 4" or method == "GPT 3.5" or method == "Gemini":
        r = LLMs()
    elif method == "spacy":
        r = spaCy()
    elif method == "None" or method == None:
        r = text
    return r


def named_entity_recognizer(text, method="spacy", output_type="list"):

    def LLMs():
        if output_type == "list":
            fixed_initial = ALL_MODELS_NER_PROMPTS.VALUE['NER_FIXED_INITIAL']
            fixed_querry = ALL_MODELS_NER_PROMPTS.VALUE['NER_FIXED_QUERRY']
        elif output_type == "dictionary":
            fixed_initial = ALL_MODELS_NER_PROMPTS.VALUE['NER_DIC_FIXED_INITIAL']
            fixed_querry = ALL_MODELS_NER_PROMPTS.VALUE['NER_DIC_FIXED_QUERRY']
        prompt = fixed_initial + text + fixed_querry
        # print(prompt)
        response = prompt_runner(method, prompt)
        # print(response)
        # remove whitespace
        return response

    def spaCy():  # to be updated
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)
        named_entities = {}
        # Iterate over named entities in the document
        for ent in doc.ents:
            # Check if entity label is in the dictionary, if not, create an empty list
            if ent.label_ not in named_entities:
                named_entities[ent.label_] = []
            # Append entity text to the corresponding key in the dictionary
            named_entities[ent.label_].append(ent.text)
            print(named_entities)
        return named_entities

    if method == "Llama" or method == "GPT 4" or method == "GPT 3.5" or method == "Gemini":
        r = LLMs()
    elif method == "spacy":
        r = spaCy()
    elif method == "None" or method == None:
        r = text
    return str(r)
