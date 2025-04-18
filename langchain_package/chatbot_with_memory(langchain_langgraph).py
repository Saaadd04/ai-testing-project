from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_mistralai import ChatMistralAI
from langgraph.graph import StateGraph, END
from typing import TypedDict, List
from langchain_core.messages import HumanMessage, AIMessage
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set Mistral API key
os.environ["MISTRAL_API_KEY"] = os.getenv("MISTRAL_API_KEY")

# Define the state structure
class ChatState(TypedDict):
    messages: List[HumanMessage | AIMessage]

# Initialize the LLM
llm = ChatMistralAI(model="mistral-large-latest", temperature=0.7)

# Define the prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful chatbot with memory. Maintain context from previous messages and provide concise, accurate responses."),
    MessagesPlaceholder(variable_name="messages"),
])

# Define the chain
chain = (
        RunnablePassthrough.assign(messages=lambda x: x["messages"])
        | prompt
        | llm
)

# Define the graph
def chatbot_node(state: ChatState) -> ChatState:
    user_message = state["messages"][-1]
    response = chain.invoke({"messages": state["messages"]})
    state["messages"].append(AIMessage(content=response.content))
    return state

# Build the graph
workflow = StateGraph(ChatState)
workflow.add_node("chatbot", chatbot_node)
workflow.set_entry_point("chatbot")
workflow.add_edge("chatbot", END)
app = workflow.compile()

# Function to interact with the chatbot
def chat_with_bot(user_input: str, state: ChatState) -> tuple[str, ChatState]:
    state["messages"].append(HumanMessage(content=user_input))
    result = app.invoke(state)
    return result["messages"][-1].content, result

# Example usage
def main():
    # # Initialize state
    # initial_state: ChatState = {"messages": []}
    #
    # # Sample conversation
    # queries = [
    #     "Hi, I'm planning a trip to Paris. Any must-visit places?",
    #     "What about good French restaurants there?",
    #     "Can you remind me what we were talking about?"
    # ]
    #
    # current_state = initial_state
    # for query in queries:
    #     print(f"\nUser: {query}")
    #     response, current_state = chat_with_bot(query, current_state)
    #     print(f"Bot: {response}")

    # Initialize state
    initial_state: ChatState = {"messages": []}
    current_state = initial_state

    print("Welcome to the ChatBot! Type your questions below (type 'exit' to quit).")

    while True:
        user_input = input("\nUser: ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        response, current_state = chat_with_bot(user_input, current_state)
        print(f"Bot: {response}")

if __name__ == "__main__":
    main()