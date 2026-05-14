# Here we are going to code that how tools are being called.
# i.e., tool calling

from dotenv import load_dotenv
load_dotenv()
from langchain_mistralai import ChatMistralAI
from langchain.tools import tool 
from langchain_core.messages import HumanMessage
from rich import print 

#1 creating a tool 
@tool
def get_text_length(text: str) -> int:
    """Returns the number of character in a given text"""
    return len(text)

tools = {
    "get_text_length" : get_text_length
}

# Initialize LLM
llm = ChatMistralAI(model = "mistral-small-2506")

#tool binding 
# Telling LLM that when you want to find length  sentence then you have a tool named get_text_length
# In the list [], you can mention 'n' number of tools
# since we have only one so we mentioned that
llm_with_tool = llm.bind_tools([get_text_length])

# After binding, the LLM knows what tools are available, but did not called them yet 
# This will only return the number what we asked, no tools called
# result = llm.invoke("Returns the number of character in a given text: Hello how are you?")

# This will also return the tools that we have since we have binded,
# Since we already have tool to get text length, so llm automatically detects the needful tools out of all the tools available for the query asked to llm
# result = llm_with_tool.invoke("Returns the number of character in a given text: Hello how are you?")

# Keeping chat history and basic input
message = []
prompt = input("You: ")
query = HumanMessage(prompt)
message.append(query)

# this wont generate any output, 
# rather it will only show the metadata, tools available etc
result = llm_with_tool.invoke(message)

message.append(result)

# if the llm is calling any tool or not, if yes then there will be a tool_calls obj in result
if result.tool_calls:
    tool_name = result.tool_calls[0]["name"]
    # This will invoke the tool with the query
    # we always execute/invoke tools manually
    # This ToolMessage will already have the output from the tool in the obj(content)
    tool_message = tools[tool_name].invoke(result.tool_calls[0])
    message.append(tool_message)
   
# This will generate the real response (i.e., the length)
# But why we need anothr llm call if we have results from the tool_message.content??

# 1. User-facing answer vs raw tool output
# The tool returns data (a number, JSON, an error string). The user asked in natural language. The second model pass is often there so the assistant can say something like: “The phrase has 17 characters,” in context, same language as the question, with short explanation if needed—not only the bare "17".

# 2. More than one tool or step
# In real flows the model may need several tool results, or to interpret them (“round up”, “compare A and B”, “if empty say …”). That reasoning is usually left to the model after all tool messages are in the thread.

# 3. Errors and edge cases
# If the tool returns an error payload, the model can apologize or retry with different args. If you only print tool_message.content, the user might see a raw stack trace or opaque blob.
result = llm_with_tool.invoke(message)
print(result.content)