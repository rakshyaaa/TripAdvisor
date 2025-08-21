import requests

# LangChain imports
from langchain.agents import AgentType, initialize_agent
from langchain.memory import ConversationBufferMemory
from langchain_ollama import OllamaLLM
from langchain_community.tools import DuckDuckGoSearchResults
from database import recommend_next_trip
from langchain.tools import Tool


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


def create_trip_advisor_agent(model_name="llama3"):
    """Creates and returns a configured trip advisor AI agent using Ollama"""

    # Initialize the local Ollama language model
    llm = OllamaLLM(
        model=model_name,
        base_url="http://localhost:11434",
        temperature=0.1,  # Low temperature for consistent, factual responses
    )

    recommend_tool = Tool(
        name="FundraisingTripAdvisor",
        func=recommend_next_trip,
        description="Suggests next fundraising visits based on the user's history on engagement score, title, and position"
    )

    search_tool = DuckDuckGoSearchResults(
        description="Use this tool to find NEW donors, foundations, or fundraising events in a given city or region. Input should be the city or organization name."
    )

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

    # system_prompt = """
    #     You are a friendly and knowledgeable AI trip advisor assistant powered by Llama3. Your role is to help users plan their perfect trips by:

    #     1. Researching destinations and providing up-to-date information
    #     2. Finding popular attractions and activities
    #     3. Suggesting accommodations based on user preferences
    #     4. Providing local transportation options
    #     5. Offering weather information and best travel times
    #     6. Giving budger estimated and travel tips

    #     Always be helpful, enthusiastic, and provide detailed, actionable advice. Use the available tools to gather information and give accurate recommendations after verifying it.

    #     When users ask questions:
    #     - Be conversational and friendly
    #     - Ask clarifying questions if needed
    #     - Use the tools to provide specific, accurate information
    #     - Give practical tips and suggestions
    #     - Consider their budget, interests, and travel style

    #     Remember: You're here to make travel planning easier and more enjoyable!
    #     """

    system_prompt = """
        You are a strategic fundraising travel advisor.
        When giving recommendations, always:

        1. ALWAYS use the 'FundraisingTripAdvisor' tool to look up the stakeholder’s past meetings and engagement scores from the database.
        2. THEN use the 'DuckDuckGoSearchResults' tool to research potential NEW places to visit or new persons/organizations to meet in those regions.
        3. Combine both insights: prioritize follow-ups with high engagement AND suggest fresh opportunities (events, new donors, or foundations).
            
        Never invent names or organizations - rely only on database results or search tool outputs.

        Verify each information and return actionable recommendations (who to visit, why, and suggested next steps).
        """

    # prompt = ChatPromptTemplate.from_messages([
    #     SystemMessagePromptTemplate.from_template(system_prompt),
    # ])

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
