import openai
from dotenv import load_dotenv
from typing_extensions import override
from openai import AssistantEventHandler
import time

# Custom event handler for processing assistant's responses
class EventHandler(AssistantEventHandler):
    def __init__(self):
        super().__init__()
        self._response_text = ""  # A buffer to accumulate the assistant's streamed text

    @override
    def on_text_created(self, text) -> None:
        # Log assistant's response start to terminal
        print(f"\nassistant > ", end="", flush=True)

    @override
    def on_text_delta(self, delta, snapshot):
        # Print incremental text output from the assistant
        print(delta.value, end="", flush=True)
        
        # Accumulate text for returning to front-end
        self._response_text += delta.value

    def on_tool_call_created(self, tool_call):
        # Log tool call type
        print(f"\nassistant > {tool_call.type}\n", flush=True)
  
    def on_tool_call_delta(self, delta, snapshot):
        # Handle streamed responses for code interpreter tool
        if delta.type == 'code_interpreter':
            if delta.code_interpreter.input:
                print(delta.code_interpreter.input, end="", flush=True)
            if delta.code_interpreter.outputs:
                print(f"\n\noutput >", flush=True)
                for output in delta.code_interpreter.outputs:
                    if output.type == "logs":
                        print(f"\n{output.logs}", flush=True)

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client
client = openai.OpenAI()

# Function to process a user message within a thread
def process_message(text, thread_id, assistant_id):
    """
    

    Args:
        text: The text content of the message
        thread_id: The thread id of the message from openai client

    Returns:
        _type_: _description_
    """
    messages = client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=text
    )
    return messages

# Function to process a message with an assistant and attached file
def process_message_with_assistant(text, thread_id, file_id):
    
    """_
    Args:
        text (str): The text content of the message
        thread_id : The thread id of the message from openai client
        file_id : The file id of the message from openai client
        
    Returns:

        message : The message from the openai clien
    """
    message = client.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=text,
            attachments=[{"file_id": file_id, 'tools':[{"type":"file_search"},{'type':"code_interpreter"}]}]
        )
    return message
   
# Function to stream assistant responses in real-time
def stream_message(thread_id, assistant_id):
    handler = EventHandler()  # Custom event handler for handling assistant's response
    with client.beta.threads.runs.stream(
        thread_id=thread_id,
        assistant_id=assistant_id,
        event_handler=handler,
    ) as stream:
        stream.until_done()
    return handler._response_text  # Return accumulated response text

# Function to simulate streaming text with incremental display
def simulate_streaming_text(full_text: str, placeholder, chunk_size=10, delay=0.03):
    displayed_text = ""
    for i in range(0, len(full_text), chunk_size):
        displayed_text += full_text[i : i + chunk_size]
        placeholder.markdown(displayed_text + "▌")  # Show cursor at the end
        time.sleep(delay)
    placeholder.markdown(displayed_text)  # Final displayed text