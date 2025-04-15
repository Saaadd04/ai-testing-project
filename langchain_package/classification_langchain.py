
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
# from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
import os

load_dotenv()

def load_api_key():
    key = os.getenv("MISTRAL_API_KEY")
    if key:
        print("API key loaded successfully.")
    else:
        raise ValueError("MISTRAL_AI API key not found in .env file.")


def get_prompt():
    return ChatPromptTemplate.from_template(
        """
        Extract the desired information from the following passage.
        Only extract the properties mentioned in the 'Classification' function.

        Passage:
        {input}
        """
    )


class Simple_Classification(BaseModel):
    sentiment: str = Field(description="The sentiment of the text")
    aggressiveness: int = Field(description="Aggression level from 1 to 10")
    language: str = Field(description="Language of the text")




def run(input_text: str):

    model_schema = Simple_Classification

    llm = ChatMistralAI(temperature=0, model="mistral-large-latest").with_structured_output(model_schema)
    prompt = get_prompt().invoke({"input": input_text})
    response = llm.invoke(prompt)

    # print(f"\n Classification Mode: {mode.upper()}")
    print("Input:", input_text)
    print("Output:", response.model_dump())

if __name__ == "__main__":
    load_api_key()

    input_text = input("Enter a text to test: ")
    # run(input_text, mode="simple")
    run(input_text)