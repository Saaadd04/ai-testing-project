import os
from mistralai import Mistral

from dotenv import load_dotenv

load_dotenv()

if __name__ == '__main__':
    api_key=os.getenv('MISTRAL_API_KEY')
    if api_key is None:
        print('You need to set your MISTRAL_API_KEY environment variable')
        exit(1)

    print(f'api key is :{api_key}')
print("-----------------------------------------")
model = "mistral-large-latest"

messages = [
    {
        "role": "system",
        "content": "My name is Syed Saad"
    },
    {
        "role": "user",
        "content": "What do you know about me ?"
    },
    # {
    #     "role":"user",
    #     "content": "What is Bhopal?"
    # }
]

client = Mistral(api_key=api_key)


# chat_response = client.chat.complete(
#     model = model,
#     messages = messages
# )
#
# print(chat_response.choices[0].message.content)

print("Streaming response:\n")
stream_response = client.chat.stream(model=model, messages=messages)
for var in stream_response:
    if var.data.choices[0].delta.content is not None:
        print(var.data.choices[0].delta.content, end='', flush=True)