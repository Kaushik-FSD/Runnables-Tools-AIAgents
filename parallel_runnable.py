from dotenv import load_dotenv
load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel,RunnableLambda

# Components
model = ChatMistralAI(model="mistral-small-2506")
parser = StrOutputParser()

# Two different prompts
short_prompt = ChatPromptTemplate.from_template(
    "Explain {topic} in 1-2 lines"
)

detailed_prompt = ChatPromptTemplate.from_template(
    "Explain {topic} in detail"
)

# Input
topic = "Machine Learning"

# parallel runnable can be mentioned in a dict
# This is the basic one
# But this will give error, since the chain var is a dict and not a runnable so it will break

# chain = {
#     "short" : short_prompt | model | parser,
#     "detailed" : detailed_prompt | model | parser
# }

# result = chain.invoke({"topic" : topic})
# print(result)

# Making the dict of type runnable, this should do the job
# chain = RunnableParallel({
#     "short" : short_prompt | model | parser,
#     "detailed" : detailed_prompt | model | parser
# })

# Here if you see we havve passed lambda function
# We need this only because we are passing different topics for different pipelines
# If its same value for both then we can skip lamba as above, but here we need it since we are passing a dict
# and from dict we are selecting the short or detailed based on the type of runnable
chain = RunnableParallel({
    "short" :RunnableLambda(lambda x :x['short']) |short_prompt | model | parser ,
    "detailed" :RunnableLambda(lambda x: x['detailed']) |detailed_prompt |model |parser
})

# if we want to pass same value to the topic we can menthion as one obj
# result = chain.invoke({"topic":"Machine Learning"})

# But if we want to pass different values to short and detailed then we pass it like this:
result = chain.invoke({
    "short" : {"topic":"Machine Learning"},
    "detailed" : {"topic":"Deep Learning"}
})

print(result['short'])
print(result['detailed'])