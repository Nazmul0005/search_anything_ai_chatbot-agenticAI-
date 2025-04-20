import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import create_react_agent
from langchain_core.messages.ai import AIMessage
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize Streamlit session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Set up the page
st.title("AI Search Assistant")

# Initialize components
@st.cache_resource
def initialize_agent():
    groq_llm = ChatGroq(model='llama-3.3-70b-versatile')
    search_tool = TavilySearchResults(max_results=2)
    system_prompt = "Act as an AI chatbot who is smart and friendly"
    
    agent = create_react_agent(
        model=groq_llm,
        tools=[search_tool],
        state_modifier=system_prompt
    )
    return agent

agent = initialize_agent()

# Chat input
user_input = st.chat_input("Ask me anything...")

# Handle user input
if user_input:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Get agent response
    state = {"messages": user_input}
    response = agent.invoke(state)
    messages = response.get("messages")
    ai_messages = [message.content for message in messages if isinstance(message, AIMessage)]
    
    if ai_messages:
        # Add AI response to chat history
        st.session_state.messages.append({"role": "assistant", "content": ai_messages[-1]})

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])