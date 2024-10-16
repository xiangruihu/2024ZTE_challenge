k = ''

k_2 = ''
# from openai import OpenAI
#
# client = OpenAI()
#
# stream = client.chat.completions.create(
#     model="gpt-3.5-turbo",
#     messages=[{"role": "user", "content": "Say this is a test"}],
#     stream=True,
# )
# for chunk in stream:
#     if chunk.choices[0].delta.content is not None:
#         print(chunk.choices[0].delta.content, end="")
import os
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'

import openai



openai.api_key = k_2
# openai.api_base = "https://openai.wndbac.cn/v1"

def get_completion(prompt, model = 'gpt-3.5-turbo'):
    messages = [{'role': 'user', 'content':prompt}]
    response = openai.ChatCompletion.create(
        model = model,
        messages = messages,
        temperature=0,
    )

    return response.choices[0].message['content']

get_completion('what is 1+1?')
