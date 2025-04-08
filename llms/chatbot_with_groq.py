import os

from groq import Groq
from dotenv import load_dotenv

load_dotenv()

if __name__ == '__main__':
    api_key = os.getenv('GROQ_API_KEY')
    if api_key is None:
        print('You need to set your GROQ_API_KEY environment variable')
        exit(1)

    print(f'api key is :{api_key}')
    print("-----------------------------------------")

    ## Groq model ("mixtral-8x7b-32768" or "llama3-70b-8192")
    model = "llama3-70b-8192"

    messages = [
        {
            "role": "system",
            "content": "My name is Syed Saad"
        },
        {
            "role": "user",
            "content": "What is a computer?"
        },
        # {
        #     "role":"user",
        #     "content": "What is Bhopal?"
        # }
    ]

    client = Groq(api_key=api_key)

    # chat_response = client.chat.complete(
    #     model = model,
    #     messages = messages
    # )
    #
    # print(chat_response.choices[0].message.content)

    stream_response = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True
    )
    for chunk in stream_response:
        if chunk.choices[0].delta.content is not None:
            print(chunk.choices[0].delta.content, end='', flush=True)
