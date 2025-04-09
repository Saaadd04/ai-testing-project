import os
from mistralai import Mistral
from dotenv import load_dotenv

load_dotenv()



class Chatbot:

    def __init__(self,api_key,model):
        self.api_key=api_key
        self.model=model
        self.conversation_history=[]
        self.mistral_client=Mistral(api_key=api_key)
        self.initialize_context()

    def initialize_context(self):
        """
        Get the profiles data from the database
        Create a String out of each row and concatenate the same into one large string
        add a new object to the conversation history
        {
        "role":"system",
        "content":"<<Large string containing details of all profiles>>"
        }

        :return:
        """
        pass


    def get_user_input(self):
        user_input=input("\nYou:")
        user_message={
            "role":"user",
            "content": user_input
        }
        self.conversation_history.append((user_message))

    def send_request(self):
        stream_response=self.mistral_client.chat.stream(
            model=self.model,
            messages=self.conversation_history
        )
        buffer=""
        for var in stream_response:
            content=var.data.choices[0].delta.content
            print(content, end='', flush=True)
            buffer+=content

        if buffer.strip():
            assistant_message={
                "role":"assistant",
                "content":buffer
            }
            self.conversation_history.append(assistant_message)


    def run(self):
        while True:
            self.get_user_input()
            self.send_request()




if __name__=="__main__":
    api_key=os.getenv('MISTRAL_API_KEY')
    if api_key is None:
        print('You need to set your MISTRAL_API_KEY environment variable')
        exit(1)

    chat_bot=Chatbot(os.getenv("MISTRAL_API_KEY"),'mistral-large-latest')
    chat_bot.run()


