
import streamlit as st
import os
from trip_agent import create_trip_advisor_agent, chat_with_agent, check_ollama_status

# Streamlit App
st.title("AI Trip Advisor")
st.subheader(
    "Your Personal Travel Planning Assistant (Powered by Local Llama3)")

# Check Ollama status
ollama_available, llama3_model = check_ollama_status()

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")

    if ollama_available:
        st.success(f"✅ Ollama is running with {llama3_model}!")
        st.info("🚀 Ready to use your local AI assistant!")
    else:
        st.error("❌ Ollama is not running")
        st.warning("Please start Ollama and pull Llama3:")
        st.code("ollama serve")
        st.code("ollama pull llama3")

# Initialize agent
if "agent" not in st.session_state and ollama_available:
    try:
        st.session_state.agent = create_trip_advisor_agent(llama3_model)
        st.success("Agent initialized!")
    except Exception as e:
        st.error(f"Error initializing agent: {e}")

# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask me about your travel plans!"):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get agent response
    if "agent" in st.session_state:
        with st.chat_message("assistant"):
            with st.spinner("Planning your trip..."):
                response = chat_with_agent(st.session_state.agent, prompt)
                st.markdown(response)
                st.session_state.messages.append(
                    {"role": "assistant", "content": response})
    else:
        st.error("Please ensure Ollama is running and Llama3 is installed!")

# Example questions
st.sidebar.header("Example Questions")
examples = [
    "I want to visit Europe in summer",
    "Plan a 5-day itinerary for Tokyo",
    "What's the best time to visit Thailand?",
    "Calculate costs for a week in Paris"
]

for example in examples:
    if st.sidebar.button(example):
        st.session_state.messages.append({"role": "user", "content": example})
        st.rerun()
