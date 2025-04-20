#step1 : api keys for groq and tvaily
import os
from dotenv import load_dotenv
load_dotenv()





#step2: set up llms
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_google_genai import ChatGoogleGenerativeAI

groq_llm=ChatGroq(
    model='llama-3.3-70b-versatile'
    )

google_llm=ChatGoogleGenerativeAI(
    model='gemini-2.0-flash-lite'
)

search_tool=TavilySearchResults(max_results=2)




#step3: set up ai agent with search tool functionality
from langgraph.prebuilt import create_react_agent
from langchain_core.messages.ai import AIMessage

system_prompt = "Act as an AI chatbot who is smart and friendly"

agent=create_react_agent(
    model=groq_llm,
    tools=[search_tool],
    state_modifier=system_prompt
)

query='What is the current dollar rate from bdt to usd and usd to bdt'
state={"messages": query}
response=agent.invoke(state)
messages=response.get("messages")
ai_messages=[message.content for message in messages if isinstance(message, AIMessage)]
print(ai_messages[-1])