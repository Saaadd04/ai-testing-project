import os
import getpass
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

# API key checks
if not os.getenv("MISTRAL_API_KEY"):
    os.environ["MISTRAL_API_KEY"] = getpass.getpass("Please enter the Mistral API key: ")
if not os.getenv("TAVILY_API_KEY"):
    os.environ["TAVILY_API_KEY"] = getpass.getpass("Please enter the Tavily API key: ")

# Initialize model and tools
language_model = init_chat_model("mistral-large-latest", model_provider="mistralai")
tavily_search_tool = TavilySearchResults(max_results=3)
tools = [tavily_search_tool]
language_model_with_tools = language_model.bind_tools(tools)

# Set up agent with memory
memory_handler = MemorySaver()
agent_executor = create_react_agent(language_model_with_tools, tools )

if __name__ == "__main__":
    print("Hello! I am a LangChain chat agent. Type 'quit' or 'exit' to exit.")

    while True:
        try:
            user_query = input("You: ").strip()
            if user_query.lower() in ["quit", "exit"]:
                print("Session ended. Goodbye!")
                break
            if not user_query:
                print("Please enter a valid query.")
                continue

            # Stream responses
            for response_step in agent_executor.stream(
                    {"messages": [HumanMessage(content=user_query)]},
                    stream_mode="values",
            ):
                try:
                    response_step["messages"][-1].pretty_print()
                except AttributeError:
                    print("Error: Unable to print response step.")

            # Final response
            final_response = agent_executor.invoke({"messages": [HumanMessage(content=user_query)]})
            print(f"Agent: {final_response['messages'][-1].content}")

            # Log tool usage
            if "tool_calls" in final_response["messages"][-1].additional_kwargs:
                print("Tools used:", final_response["messages"][-1].additional_kwargs["tool_calls"])

        except KeyboardInterrupt:
            print("\nSession interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")