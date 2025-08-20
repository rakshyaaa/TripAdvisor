import os
from datetime import datetime
from typing import List, Dict, Any, Optional
import requests

# LangChain imports
from langchain.agents import AgentType, initialize_agent, Tool
from langchain.memory import ConversationBufferMemory
from langchain.prompts import SystemMessagePromptTemplate, ChatPromptTemplate
from langchain.schema import BaseOutputParser
from langchain.tools import BaseTool
from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain_ollama import OllamaLLM
from pydantic import BaseModel, Field
from langchain_community.tools import DuckDuckGoSearchResults


def check_ollama_status():
    """Check if Ollama is running and has Llama3 model available"""
    try:
        # Check if Ollama server is running
        response = requests.get("http://localhost:11434/api/tags", timeout=5)

        if response.status_code == 200:
            models = response.json()

            model_names = [model['name'] for model in models.get('models', [])]

            # Check for Llama3 variants
            llama3_models = [
                name for name in model_names if 'llama3' in name.lower()]
            print(llama3_models)
            if llama3_models:
                # Return first available Llama3 model
                return True, llama3_models[0]
            else:
                return False, None
        else:
            return False, None
    except requests.exceptions.ConnectionError:
        return False, None
    except Exception:
        return False, None


def create_trip_advisor_agent(model_name="llama3"):
    """Creates and returns a configured trip advisor AI agent using Ollama"""

    # Initialize the local Ollama language model
    llm = OllamaLLM(
        model=model_name,
        base_url="http://localhost:11434",
        temperature=0.1,  # Low temperature for consistent, factual responses
    )

    # Create tool instances
    tools = [
        DuckDuckGoSearchResults()
    ]

    # Create memory for conversation history
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    system_prompt = """
        You are a friendly and knowledgeable AI trip advisor assistant powered by Llama3. Your role is to help users plan their perfect trips by:

        1. Researching destinations and providing up-to-date information
        2. Finding popular attractions and activities
        3. Suggesting accommodations based on user preferences
        4. Providing local transportation options
        5. Offering weather information and best travel times
        6. Giving budger estimated and travel tips

        Always be helpful, enthusiastic, and provide detailed, actionable advice. Use the available tools to gather information and give accurate recommendations after verifying it.

        When users ask questions:
        - Be conversational and friendly
        - Ask clarifying questions if needed
        - Use the tools to provide specific, accurate information
        - Give practical tips and suggestions
        - Consider their budget, interests, and travel style

        Remember: You're here to make travel planning easier and more enjoyable!
        """
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_prompt),
    ])

    # Create the agent with tools, memory and custom_system_prompt
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True,
        agent_kwargs={"system_message": system_prompt}  # prompt injection
    )

    print(agent)
    return agent


def chat_with_agent(agent, message):
    """Function to chat with the trip advisor agent"""
    if agent is None:
        return "Please create the agent first by ensuring Ollama is running!"

    try:
        response = agent.invoke({"input": message})
        return response.get("output", "No response received")
    except Exception as e:
        return f"Error: {e}"
