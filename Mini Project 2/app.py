# Import the necessary libraries
import streamlit as st
from openai import OpenAI  # Install the OpenAI library using pip install openai
import os
from dotenv import load_dotenv

# 加载.env文件中的环境变量
load_dotenv('/home/sky/projects/Win25_LLM/Mini Project 2/.env')

# Set the title of the Streamlit app
st.title("Mini Project 2: Streamlit Chatbot")

# Initialize the OpenAI client with your API key from environment variable
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Define a function to get the conversation history (Not required for Part-2, will be useful in Part-3)
def get_conversation() -> str:
    """
    Returns a formatted string representation of the conversation history.
    """
    conversation = ""
    for message in st.session_state.messages:
        role = message["role"]
        content = message["content"]
        conversation += f"{role.capitalize()}: {content}\n"
    return conversation

# Check for existing session state variables and initialize them if they don't exist
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-4o-mini"  # Set the model to GPT-4o-mini

if "messages" not in st.session_state:
    st.session_state.messages = []  # Initialize an empty list to store messages

# Display existing chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):  # Display the message based on the role (user or assistant)
        st.markdown(message["content"])  # Render the message content

# Wait for user input
if prompt := st.chat_input("What would you like to chat about?"):
    # Append the user's message to the session state's messages list
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display the user's message in the chat interface
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate the assistant's response using OpenAI API
    with st.chat_message("assistant"):
        # Send the conversation history to OpenAI API
        response = client.chat.completions.create(
            model=st.session_state["openai_model"],
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ]
        )
        # Get the assistant's response from the API response
        assistant_response = response.choices[0].message.content
        st.markdown(assistant_response)  # Display the assistant's response

    # Append the assistant's response to the session state's messages list
    st.session_state.messages.append({"role": "assistant", "content": assistant_response})