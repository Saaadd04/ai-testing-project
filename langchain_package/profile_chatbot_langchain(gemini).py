from pymongo import MongoClient
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.chat_models import init_chat_model
from google.generativeai import configure
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Fetch environment variables
MONGO_CONNECTION_STRING = os.getenv("MONGO_CONNECTION_STRING")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

def fetch_profiles():
    """Fetch profile data from MongoDB 'folders' collection."""
    client = MongoClient(MONGO_CONNECTION_STRING)
    db = client['app-dev']
    collection = db['profiles']
    documents = collection.find()

    large_string = ""


    for doc in documents:
        full_name = f"{doc.get('firstName', 'N/A')} {doc.get('lastName', 'N/A')}".strip()
        expertise = doc.get('areaOfExpertise', 'N/A')

        # Build the profile string
        profile_string = f"Name: {full_name}, areaOfExpertise: {doc.get('areaOfExpertise')}, currentLocation: {doc.get('currentLocation')}, Experience: {doc.get('experience')}, highlightedSkills: {doc.get('highlightedSkills')} "
        large_string += f"{profile_string}\n"


    client.close()
    return large_string

if __name__ == "__main__":
    # Configure Gemini API
    configure(api_key=GEMINI_API_KEY)

    # Fetch profile data
    profile_data = fetch_profiles()
    # print("Profile Data:", profile_data)

    # Define the prompt template
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are a chatbot for the Hiretallent project. Below is a list of profiles from our database. Use this data to answer the user's query.\n\nProfiles:\n{profiles}\n"),
        ("human", "User Query: {query}")
    ])

    # Initialize the Gemini model with LangChain
    chat_model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GEMINI_API_KEY)

    # Create the chain
    chain = prompt_template | chat_model

    # Interactive loop
    while True:
        query = input("Ask a question ?(or type 'exit' to stop): ")
        if query.lower() == 'exit':
            print("You have successfully exited.\nre-run the program if you want to use again! ")
            break
        response = chain.invoke({"profiles": profile_data, "query": query})
        print("Gemini Response:", response.content)
        print("\nThankyou! I hope your query was answered.\n")