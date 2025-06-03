# config Gemini

# !pip install -q -U google-generativeai

# import pathlib
# import textwrap
# import os
# import google.generativeai as genai
# from langchain_google_genai import ChatGoogleGenerativeAI
# from google.cloud.aiplatform_v1beta1.types.content import SafetySetting
# from vertexai.preview.generative_models import HarmCategory, HarmBlockThreshold
# from constants.api import *


# os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY
# # genai.configure(api_key=GOOGLE_API_KEY)
# # gemini_model = genai.GenerativeModel('gemini-pro')

# # def to_markdown(text):
# #   text = text.replace('•', '  *')
# #   return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

# safety_settings = [
#     {
#         "category": HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,        "threshold": HarmBlockThreshold.BLOCK_NONE, },    {
#         "category": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,        "threshold": HarmBlockThreshold.BLOCK_NONE, },    {
#         "category": HarmCategory.HARM_CATEGORY_HATE_SPEECH,        "threshold": HarmBlockThreshold.BLOCK_NONE, },    {
#         "category": HarmCategory.HARM_CATEGORY_HARASSMENT,        "threshold": HarmBlockThreshold.BLOCK_NONE, },]


# llm = ChatGoogleGenerativeAI(model="gemini-pro")


# def gemini(msg):
#     # response = gemini_model.generate_content(msg)
#     # return response.text, response.prompt_feedback
#     result = llm.invoke(msg)
#     return result.content
