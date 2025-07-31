# Trip Advisor AI Agent using LangChain + Ollama (Local Llama3)
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
import requests

# LangChain imports
from langchain.agents import AgentType, initialize_agent, Tool
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.schema import BaseOutputParser
from langchain.tools import BaseTool
from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain_ollama import OllamaLLM
from pydantic import BaseModel, Field

# Custom Tools for Trip Planning
class DestinationRecommenderTool(BaseTool):
    """Tool to recommend destinations based on preferences"""
    name: str = "destination_recommender"
    description: str = "Recommends travel destinations based on user preferences like budget, interests, season, etc."
    
    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        # This is a simplified implementation - in reality, you'd connect to a travel database or API
        preferences = query.lower()
        
        recommendations = []
        
        if "beach" in preferences or "tropical" in preferences:
            recommendations.extend(["Maldives", "Bali", "Caribbean Islands", "Hawaii"])
        if "mountain" in preferences or "hiking" in preferences:
            recommendations.extend(["Swiss Alps", "Nepal", "Patagonia", "Rocky Mountains"])
        if "culture" in preferences or "history" in preferences:
            recommendations.extend(["Rome", "Kyoto", "Istanbul", "Cairo"])
        if "budget" in preferences or "cheap" in preferences:
            recommendations.extend(["Thailand", "Vietnam", "India", "Eastern Europe"])
        if "luxury" in preferences or "expensive" in preferences:
            recommendations.extend(["Monaco", "Dubai", "Tokyo", "New York"])
        if "adventure" in preferences:
            recommendations.extend(["New Zealand", "Costa Rica", "Iceland", "Patagonia"])
        
        if not recommendations:
            recommendations = ["Paris", "London", "Barcelona", "Amsterdam", "Prague"]
        
        return f"Based on your preferences '{query}', I recommend these destinations: {', '.join(recommendations[:5])}"

class ItineraryPlannerTool(BaseTool):
    """Tool to create travel itineraries"""
    name: str = "itinerary_planner"
    description: str = "Creates detailed travel itineraries for specific destinations including activities, restaurants, and timing."
    
    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        # Parse the destination and duration from query
        destination = query.split(" for ")[0] if " for " in query else query
        
        # Sample itinerary data (in a real app, this would come from a database or API)
        itineraries = {
            "paris": {
                "day1": "Morning: Visit Eiffel Tower, Afternoon: Louvre Museum, Evening: Seine River Cruise",
                "day2": "Morning: Notre-Dame Cathedral, Afternoon: Montmartre & Sacré-Cœur, Evening: Latin Quarter dinner",
                "day3": "Morning: Palace of Versailles, Afternoon: Champs-Élysées shopping, Evening: Moulin Rouge show"
            },
            "rome": {
                "day1": "Morning: Colosseum tour, Afternoon: Roman Forum, Evening: Trastevere dinner",
                "day2": "Morning: Vatican Museums & Sistine Chapel, Afternoon: St. Peter's Basilica, Evening: Spanish Steps",
                "day3": "Morning: Pantheon, Afternoon: Trevi Fountain & Villa Borghese, Evening: Campo de' Fiori"
            },
            "tokyo": {
                "day1": "Morning: Senso-ji Temple, Afternoon: Tokyo Skytree, Evening: Shibuya Crossing",
                "day2": "Morning: Tsukiji Fish Market, Afternoon: Imperial Palace, Evening: Ginza district",
                "day3": "Morning: Meiji Shrine, Afternoon: Harajuku & Omotesando, Evening: Robot Restaurant"
            }
        }
        
        dest_key = destination.lower().replace(" ", "")
        if dest_key in itineraries:
            itinerary = itineraries[dest_key]
            result = f"Here's a 3-day itinerary for {destination}:\n\n"
            for day, activities in itinerary.items():
                result += f"{day.upper()}: {activities}\n"
            return result
        else:
            return f"I don't have a specific itinerary for {destination} yet, but I can help you plan based on popular attractions and activities there!"

class BudgetCalculatorTool(BaseTool):
    """Tool to estimate travel costs"""
    name: str = "budget_calculator"
    description: str = "Calculates estimated travel costs including flights, accommodation, food, and activities."
    
    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        # Sample budget estimates (in reality, you'd use real-time pricing APIs)
        budgets = {
            "paris": {"flights": 800, "hotel": 150, "food": 60, "activities": 40},
            "rome": {"flights": 700, "hotel": 120, "food": 45, "activities": 35},
            "tokyo": {"flights": 1200, "hotel": 180, "food": 80, "activities": 50},
            "thailand": {"flights": 900, "hotel": 40, "food": 15, "activities": 25},
            "new york": {"flights": 500, "hotel": 250, "food": 70, "activities": 60}
        }
        
        destination = query.lower().replace(" ", "")
        days = 7  # Default to 7 days
        
        # Try to extract number of days from query
        import re
        day_match = re.search(r'(\d+)\s*days?', query.lower())
        if day_match:
            days = int(day_match.group(1))
        
        if destination in budgets:
            budget = budgets[destination]
            total_per_day = budget["hotel"] + budget["food"] + budget["activities"]
            total_cost = budget["flights"] + (total_per_day * days)
            
            return f"""Budget estimate for {destination.title()} ({days} days):
            
Flight: ${budget['flights']}
Hotel: ${budget['hotel']}/night × {days} nights = ${budget['hotel'] * days}
Food: ${budget['food']}/day × {days} days = ${budget['food'] * days}
Activities: ${budget['activities']}/day × {days} days = ${budget['activities'] * days}

Total estimated cost: ${total_cost}
Daily cost (excluding flights): ${total_per_day}"""
        else:
            return f"I don't have specific budget data for {destination}, but typical costs vary widely by destination. Budget destinations: $50-100/day, Mid-range: $100-200/day, Luxury: $300+/day"

class WeatherInfoTool(BaseTool):
    """Tool to provide weather and seasonal information"""
    name: str = "weather_info"
    description: str = "Provides weather information and best times to visit destinations."
    
    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        destination = query.lower()
        
        # Sample weather data (in reality, you'd use a weather API)
        weather_data = {
            "paris": {
                "best_months": "April-June, September-October",
                "climate": "Temperate oceanic climate",
                "summer": "Warm (15-25°C), can be rainy",
                "winter": "Cool (3-7°C), occasional snow"
            },
            "tokyo": {
                "best_months": "March-May, September-November", 
                "climate": "Humid subtropical climate",
                "summer": "Hot and humid (25-30°C), rainy season June-July",
                "winter": "Cool and dry (5-10°C)"
            },
            "thailand": {
                "best_months": "November-March",
                "climate": "Tropical monsoon climate",
                "summer": "Hot (28-35°C), rainy season May-October",
                "winter": "Warm and dry (25-30°C)"
            }
        }
        
        dest_key = destination.replace(" ", "").lower()
        if dest_key in weather_data:
            data = weather_data[dest_key]
            return f"""Weather information for {destination.title()}:

Climate: {data['climate']}
Best months to visit: {data['best_months']}
Summer: {data['summer']}
Winter: {data['winter']}

Remember to check current weather forecasts before traveling!"""
        else:
            return f"I don't have specific weather data for {destination}, but I recommend checking current weather forecasts and seasonal patterns before planning your trip."

class TravelTipsTool(BaseTool):
    """Tool to provide travel tips and cultural information"""
    name: str = "travel_tips"
    description: str = "Provides travel tips, cultural etiquette, and practical advice for destinations."
    
    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        destination = query.lower()
        
        tips_data = {
            "japan": [
                "Remove shoes when entering homes and some restaurants",
                "Bow as a greeting instead of shaking hands",
                "Don't tip - it's considered rude",
                "Learn basic Japanese phrases",
                "Get a JR Pass for train travel",
                "Cash is preferred over cards in many places"
            ],
            "thailand": [
                "Dress modestly when visiting temples",
                "Don't touch someone's head",
                "Remove shoes before entering temples and homes",
                "Bargain at markets but not in malls",
                "Try street food - it's usually safe and delicious",
                "Learn basic Thai greetings"
            ],
            "france": [
                "Learn basic French phrases - locals appreciate the effort",
                "Dress well, especially in Paris",
                "Greet shopkeepers when entering stores",
                "Lunch is typically 12-2pm, dinner after 7:30pm",
                "Tipping 10% is sufficient in restaurants",
                "Many museums are free on first Sunday mornings"
            ]
        }
        
        dest_key = destination.replace(" ", "").lower()
        if dest_key in tips_data:
            tips = tips_data[dest_key]
            result = f"Travel tips for {destination.title()}:\n\n"
            for i, tip in enumerate(tips, 1):
                result += f"{i}. {tip}\n"
            return result
        else:
            return "General travel tips: Research local customs, learn basic phrases, respect dress codes, try local cuisine, and always have travel insurance!"

def check_ollama_status():
    """Check if Ollama is running and has Llama3 model available"""
    try:
        # Check if Ollama server is running
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json()
            model_names = [model['name'] for model in models.get('models', [])]
            
            # Check for Llama3 variants
            llama3_models = [name for name in model_names if 'llama3' in name.lower()]
            
            if llama3_models:
                return True, llama3_models[0]  # Return first available Llama3 model
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
        DestinationRecommenderTool(),
        ItineraryPlannerTool(),
        BudgetCalculatorTool(),
        WeatherInfoTool(),
        TravelTipsTool()
    ]
    
    # Create memory for conversation history
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    
    # Create the agent with tools and memory
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True
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
