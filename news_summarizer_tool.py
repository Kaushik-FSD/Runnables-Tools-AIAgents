# This code fetches news and summarizes top 5 news from browser
from dotenv import load_dotenv
load_dotenv()
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# limiting only 5 results
# This is Tavely call, it is a browser that will call/fetch data based on yout prompt
search_tool = TavilySearchResults(max_result = 5)

llm = ChatMistralAI(model = "mistral-small-2506")

prompt = ChatPromptTemplate.from_template(
    """
You are a helpful assistant

summarize the following news into clear bullet points

{news}
"""
)

chain = prompt | llm | StrOutputParser()

# This calls a web search API i.e., Tavily's web search API.
# It brings results and that we are feeding to LLM
news_result = search_tool.run("Latest AI news of 2026 ")

result = chain.invoke({"news" : news_result})

print(result)


print(search_tool.description)
print(search_tool.name)
print(search_tool.args)