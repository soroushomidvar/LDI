# config Llama
# !pip install replicate
# import os

# os.environ["REPLICATE_API_TOKEN"] is used by default


# def llama(msg):
#     output = replicate.run("meta/llama-2-70b-chat", input={"prompt": msg})
#     r = ""
#     for item in output:
#         r += item

#     return r.replace("\n", " ")
