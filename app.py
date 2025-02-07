import streamlit as st
import time
import os

# Remove SSL certificate file from environment variables if it exists
if os.environ.get("SSL_CERT_FILE"):
    del os.environ["SSL_CERT_FILE"]

from dotenv import load_dotenv
import openai

# Importing assistant-related functions from external modules
from assistant_creation import create_assistant, load_assistant, create_thread, change_assistant_model

# Importing message processing functions
from message_processing import (
    process_message,
    process_message_with_assistant,
    stream_message
)

def init_session_state():
    """Initialize Streamlit session state variables."""
    if "assistant" not in st.session_state:
        st.session_state.assistant = None  # Store assistant object
    if "thread" not in st.session_state:
        st.session_state.thread = None  # Store thread object
    if "page" not in st.session_state:
        st.session_state.page = "setup"  # Default page
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []  # Store chat messages
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = {}  # Store uploaded file metadata
    if "file_processed" not in st.session_state:
        st.session_state.file_processed = False  # Track file processing status
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = None  # Track selected AI model

def setup_openai_api():
    """Load OpenAI API key from environment variables."""
    load_dotenv()

def simulate_streaming_text(full_text: str, placeholder, chunk_size=10, delay=0.03):
    """
    Simulates streaming text display in Streamlit.

    Args:
        full_text (str): The full response text to be streamed.
        placeholder: Streamlit placeholder element for updating text.
        chunk_size (int, optional): Number of characters to display at a time. Default is 10.
        delay (float, optional): Delay in seconds between each chunk. Default is 0.03.
    """
    displayed_text = ""
    for i in range(0, len(full_text), chunk_size):
        displayed_text += full_text[i : i + chunk_size]
        placeholder.markdown(displayed_text + "▌")  # Display live text with cursor effect
        time.sleep(delay)
    placeholder.markdown(displayed_text)  # Final update without cursor

def assistant_setup_page():
    """Page for setting up or loading an assistant."""
    st.title("Assistant Setup")

    # Load existing assistant
    st.subheader("Load Existing Assistant")
    assistant_id = st.text_input("Enter Assistant ID")
    if st.button("Load Assistant"):
        if assistant_id:
            try:
                st.session_state.assistant = load_assistant(assistant_id)
                st.session_state.thread = create_thread()
                st.session_state.page = "chat"  # Navigate to chat page
            except Exception as e:
                st.error(f"Failed to load assistant: {e}")
        else:
            st.error("Please enter a valid Assistant ID.")

    # Create new assistant
    st.subheader("Create New Assistant")
    assistant_instructions = st.text_area("Enter Instructions for the New Assistant")
    if st.button("Create Assistant"):
        if assistant_instructions.strip():
            try:
                st.session_state.assistant = create_assistant(assistant_instructions)
                st.session_state.thread = create_thread()
                st.session_state.page = "chat"  # Navigate to chat page
            except Exception as e:
                st.error(f"Failed to create assistant: {e}")
        else:
            st.error("Please provide instructions to create the assistant.")

def chat_interface_page():
    """Page for interacting with the AI assistant via chat."""
    st.title("Assistant Chat Interface")

    # Sidebar for model selection and file upload
    with st.sidebar:
        st.header("Model Selection")
        available_models = ["gpt-4o", "gpt-4o-mini"]
        selected_model = st.selectbox("Select a Model", options=available_models, index=available_models.index("gpt-4o-mini"))

        # Update assistant model when user clicks the button
        if st.button("Update Model"):
            if st.session_state.assistant:
                try:
                    updated_assistant = change_assistant_model(
                        assistant_id=st.session_state.assistant.id, 
                        model=selected_model
                    )
                    st.session_state.assistant = updated_assistant  # Update stored assistant
                    st.success(f"Model updated to '{selected_model}' successfully!")
                except Exception as e:
                    st.error(f"Failed to update model: {e}")
            else:
                st.error("No assistant loaded. Please load or create an assistant first.")

        # File upload section
        st.header("File Upload")
        uploaded_file = st.file_uploader("Upload a file", type=["pdf"])

        if uploaded_file is not None:
            file_name = uploaded_file.name
            if st.session_state.file_processed and file_name in st.session_state.uploaded_files:
                st.info("Answering Questions mode.")  # Indicate the file is ready for Q&A
            else:
                # Save file temporarily
                os.makedirs("temp_uploads", exist_ok=True)
                temp_file_path = os.path.join("temp_uploads", file_name)
                with open(temp_file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Upload file to OpenAI
                client = openai.OpenAI()
                with open(temp_file_path, "rb") as file:
                    openai_file_obj = client.files.create(file=file, purpose="assistants")

                # Wait for OpenAI to process the file
                while True:
                    file_status = client.files.retrieve(openai_file_obj.id).status
                    if file_status == "processed":
                        break
                    time.sleep(1)

                # Store file ID in session state
                st.session_state.uploaded_files[file_name] = openai_file_obj.id

                # Inform assistant about the uploaded file
                process_message_with_assistant(
                    f"Here is a file: {file_name}.",
                    thread_id=st.session_state.thread.id,
                    file_id=openai_file_obj.id
                )

                # Notify the user
                st.session_state.chat_history.append(("user", f"Here is a file: {file_name}. Keep it."))
                st.session_state.chat_history.append(("assistant", "Got it! I have processed the file. You can ask me questions about it."))
                st.success(f"File '{file_name}' processed successfully!")

                # Mark file as processed
                st.session_state.file_processed = True

    # Display chat history
    for role, content in st.session_state.chat_history:
        with st.chat_message(role):
            st.write(content)

    # Chat input section
    user_input = st.chat_input("Type your message here...")
    if user_input:
        with st.chat_message("user"):
            st.write(user_input)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            placeholder.markdown("Thinking...")

        # Process user input
        process_message(
            user_input,
            thread_id=st.session_state.thread.id,
            assistant_id=st.session_state.assistant.id
        )

        # Get assistant's response
        assistant_response = stream_message(
            thread_id=st.session_state.thread.id,
            assistant_id=st.session_state.assistant.id
        )

        # Stream assistant's response
        simulate_streaming_text(assistant_response, placeholder)

        # Store messages in chat history
        st.session_state.chat_history.append(("user", user_input))
        st.session_state.chat_history.append(("assistant", assistant_response))

def main():
    """Main function to set up the Streamlit app and render pages dynamically."""
    st.set_page_config(page_title="ChatGPT App", layout="wide")
    init_session_state()
    setup_openai_api()

    # Render the appropriate page based on session state
    if st.session_state.page == "setup":
        assistant_setup_page()  # Assistant setup page
    elif st.session_state.page == "chat":
        chat_interface_page()  # Chat interface page

# Entry point for running the script
if __name__ == "__main__":
    main()
