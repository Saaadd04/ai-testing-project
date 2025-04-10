import os
from pymongo import MongoClient
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
# from langchain_mistralai import ChatMistralAI
from langchain.chat_models import init_chat_model

# Load environment variables from .env file
load_dotenv()

class Chatbot:
    def __init__(self, api_key, model):
        self.api_key = api_key
        self.model = model
        self.conversation_history = []
        # Initialize LangChain Mistral chat model
        self.mistral_client = init_chat_model(api_key = api_key, model = model)
        self.profile_data = self.fetch_profiles()
        self.initialize_context()

    def fetch_profiles(self):
        """Fetch profile data from MongoDB 'folders' collection."""
        # Use the connection string from .env
        connection_string = os.getenv("MONGO_CONNECTION_STRING")
        if not connection_string:
            raise ValueError("MONGO_CONNECTION_STRING not set in .env")

        client = MongoClient(connection_string)
        db = client['app-dev']
        collection = db['profiles']  # Updated to 'profiles' as per your code
        documents = collection.find()

        large_string = ""
        for doc in documents:
            full_name = f"{doc.get('firstName', 'N/A')} {doc.get('lastName', 'N/A')}".strip()


            # Build the profile string
            profile_string = f"Name: {full_name}, areaOfExpertise: {doc.get('areaOfExpertise')},CareerSummary:{doc.get('carrierSummary')} , highlightedSkills: {doc.get('highlightedSkills')} "
            large_string += f"{profile_string}\n"

        # Close the connection
        client.close()
        return large_string

    def initialize_context(self):
        """Set up the LangChain prompt template with profile data."""
        # Define the prompt template
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", "You are a chatbot for the Hiretallent project. Use this data to answer the user's query.\n\nProfiles:\n{profiles}\n"),
            ("human", "User Query: {query}")
        ])
        # Create the chain with the Mistral model
        self.chain = self.prompt_template | self.mistral_client
        # Add initial system message to conversation history (optional, for tracking)
        self.conversation_history.append({
            "role": "system",
            "content": f"You are a chatbot for my project. Use this data to answer the user's query.\n{self.profile_data}"
        })

    def get_user_input(self):
        """Get user input and add it to the conversation history."""
        user_input = input("\nAsk a question ?(or type 'exit' to stop):")
        user_message = {
            "role": "user",
            "content": user_input
        }
        self.conversation_history.append(user_message)
        return user_input

    def send_request(self):
        """Send the query to the LangChain chain and get the response."""
        # Use the LangChain chain to process the query with profile data
        response = self.chain.invoke({
            "profiles": self.profile_data,
            "query": self.conversation_history[-1]["content"]
        })
        print(response.content,end='', flush=True)

        # Add assistant response to conversation history
        assistant_message = {
            "role": "assistant",
            "content": response.content
        }
        self.conversation_history.append(assistant_message)

    def run(self):
        """Run the chatbot in an interactive loop."""
        while True:
            query = self.get_user_input()
            if query.lower() == 'exit':
                print("You have successfully exited.\nre-run the program if you want to use again! ")
                break
            self.send_request()
            print("\nThankyou! I hope your query was answered.\n")

if __name__ == "__main__":
    api_key = os.getenv('MISTRAL_API_KEY')
    if api_key is None:
        print('You need to set your MISTRAL_API_KEY environment variable')
        exit(1)

    chat_bot = Chatbot(api_key=os.getenv("MISTRAL_API_KEY"), model='mistral-large-latest')
    chat_bot.run()