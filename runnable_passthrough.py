from dotenv import load_dotenv
load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough


model = ChatMistralAI(model="mistral-small-2506")
parser = StrOutputParser()

code_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a code generator"),
    ("human", "{topic}")
])

explain_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant who explains code in simple terms"),
    ("human", "Explain the following code in simple words:\n{code}")
])

# We are trying to call 2 prompts: one to generate code, another to explain the code that is generated.

# This is one of the way where you can pass the o/p of one item to other
# but this is lengthy and can miss a step (error prone)
# Most importantly I wanted the code output from first model call, but I cant since its passed to next step.
# seq = code_prompt | model | parser | explain_prompt | model | parser

# This is where passthrough comes in
# This will give us the code after running prompt 1
seq = code_prompt | model | parser 


# RunnablePassthrough() returns the input only that we pased it
seq2 = RunnableParallel(
    {
        # This will hold the o/p of seq (i.e., the code)
        "code" :  RunnablePassthrough(),
        # And from the RunablePassthrough it is going to pass to explain_prompt (i.e., the code again)
        "explanation" : explain_prompt | model | parser
    }
) 

chain = seq | seq2

result = chain.invoke({"topic" : "please write a code of palindrome in python "})

print(result['code'])
print(result['explanation'])