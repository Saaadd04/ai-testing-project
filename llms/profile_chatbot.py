import os
from mistralai import Mistral
from dotenv import load_dotenv
from pymongo import MongoClient

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


        # Replace with your actual MongoDB connection string
        connection_string = "mongodb+srv://hiretalent-dev:Yulwbmn87x92EQ0U@hiretalent.doscksq.mongodb.net"

        # Connect to MongoDB
        client = MongoClient(connection_string)

        # Access the 'app-dev' database
        db = client['app-dev']

        # Access the 'folders' collection
        collection = db['profiles']

        # Fetch all documents from the 'folders' collection
        documents = collection.find()

        # Create a concatenated string from the documents
        large_string = ""
        for doc in documents:
            full_name = f"{doc.get('firstName', 'N/A')} {doc.get('lastName', 'N/A')}".strip()


            # Build the profile string
            profile_string = f"Name: {full_name}, areaOfExpertise: {doc.get('areaOfExpertise')},CareerSummary:{doc.get('carrierSummary')} , highlightedSkills: {doc.get('highlightedSkills')} "
            large_string += f"{profile_string}\n"

        # Close the connection
        client.close()

        new_message={
            "role":"system",
            "content": large_string
        }
        self.conversation_history.append((new_message))
        # print("All Profiles:", large_string)



    def get_user_input(self):
        user_input=input("\nYour Question :")
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


