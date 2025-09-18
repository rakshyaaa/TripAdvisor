import requests

# LangChain imports
from langchain.agents import AgentType, initialize_agent
from langchain.memory import ConversationBufferMemory
from langchain_ollama import OllamaLLM
from langchain_community.tools import DuckDuckGoSearchResults
from database import fetch_candidates
from langchain.tools import StructuredTool


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
            # print(llama3_models)
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


def create_trip_advisor_agent(model_name="gpt-oss:20b"):
    """Creates and returns a configured trip advisor AI agent using Ollama"""

    # Initialize the local Ollama language model
    llm = OllamaLLM(
        model=model_name,
        base_url="http://localhost:11434",
        temperature=0.1,  # Low temperature for consistent, factual responses
    )

    recommend_tool = StructuredTool.from_function(
        func=lambda args: rank_backend(**args.dict()),
        name="FundraisingTripAdvisor",
        description="Return TOP-N prioritized prospects from SQL Server (no itinerary). Use for prioritization.",

    )

    # search_tool = DuckDuckGoSearchResults(
    #     description="Use this tool to find NEW prospects, foundations, or fundraising events in a given city or region. Input should be the city or prospect's name."
    # )

    # general_questions_tool = Tool(
    #     name="GeneralQuestionsAssistant",
    #     func=lambda query: llm.invoke(query),
    #     description="Use this tool to answer general questions not related to wealth engine, travel or stakeholder engagement."
    # )

    # Create tool instances
    tools = [
        search_tool,
        recommend_tool
    ]

    # Create memory for conversation history
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    system_prompt = """
        You are Fundraising Prospect Ranker.

        Mission:
        - Recommend the next prospects to prioritize. Do NOT create itineraries or dates.

        Ground truth:
        - Use ONLY the FundraisingTripAdvisor tool for prospect facts and rankings.
        - Use WebSearch only for discovering NEW prospects/foundations/events outside our data.
        - Never invent names, gifts, or scores. Say when data is missing.

        Policy:
        1) For prioritization, call FundraisingTripAdvisor first.
        2) For discovery, call WebSearch.
        3) No travel plans.

        Output:
        - Title: Top prospects (N)
        - Then N lines:
          - name_or_id (or display name) — city, location — final_score X.XX
          - Why: 2–3 short reasons (capacity / propensity / engagement / recency)
        - Notes: mention defaults/missing fields if any.

        Guardrails:
        - Exclude Do-Not-Contact; avoid contacts within 90 days unless user overrides.
        - Always show final_score. Keep it concise.
        """

    # Create the agent with tools, memory and custom_system_prompt
    fundraising_agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        memory=memory,
        verbose=True,  # it shows what the agent is thinking/doing behind the scenes
        handle_parsing_errors=True,
        agent_kwargs={"system_message": system_prompt}  # prompt injection
    )

    return fundraising_agent


def chat_with_agent(agent, message):
    """Function to chat with the trip advisor agent"""
    if agent is None:
        return "Please create the agent first by ensuring Ollama is running!"

    try:
        response = agent.invoke({"input": message})
        return response.get("output", "No response received")
    except Exception as e:
        return f"Error: {e}"
