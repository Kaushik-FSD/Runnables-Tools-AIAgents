from langchain.tools import tool

# Custom tools are nothing but functions

@tool #decorator for creating tool 
def get_greeting(name : str) -> str: #type hints
    """Generate a greeting message for a user""" #docstring

    return f"Hello {name}, Welcome to the AI world"


# since tools are also runnables so we can call the invoke method
result = get_greeting.invoke({"name":"akarsh"})
print(result)

print(get_greeting.name)
print(get_greeting.description)
print(get_greeting.args)