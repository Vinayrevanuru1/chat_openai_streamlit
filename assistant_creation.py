import openai
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# Initialize OpenAI client
client = openai.OpenAI()

def generate_assistant_name(instruction):
    """
    Generates a name for an AI assistant based on a given instruction.
    
    Args:
        instruction (str): A description of the AI assistant.
    
    Returns:
        str: A generated assistant name.
    """
    prompt = f"Generate a suitable name for an AI assistant based on the following instruction: {instruction}, return only the name without double quotes"
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "You are an AI assistant name generator."},
                  {"role": "user", "content": prompt}],
        max_tokens=10,
        temperature=0.1
    )
    
    return response.choices[0].message.content.strip()

def change_assistant_model(assistant_id, model):
    """
    Updates the model of an existing assistant.
    
    Args:
        assistant_id (str): The ID of the assistant to update.
        model (str): The new model to assign (e.g., "gpt-4o-mini" or "gpt-4o").
    
    Returns:
        object: The updated assistant.
    """
    assistant = client.beta.assistants.update(
        assistant_id=assistant_id,
        model=model
    )
    
    return assistant

def create_assistant(instructions):
    """
    Creates a new AI assistant with the specified instructions.
    
    Args:
        instructions (str): The instructions that define the assistant's behavior.
    
    Returns:
        object: The created assistant.
    """
    assistant = client.beta.assistants.create(
        name=generate_assistant_name(instructions),
        instructions=instructions,
        model="gpt-4o-mini",
        tools=[{"type": "file_search"}, {'type': "code_interpreter"}],
    )
    
    return assistant

def load_assistant(assistant_id):
    """
    Retrieves an existing assistant by its ID.
    
    Args:
        assistant_id (str): The ID of the assistant to retrieve.
    
    Returns:
        object: The retrieved assistant.
    """
    assistant = client.beta.assistants.retrieve(assistant_id=assistant_id)
    return assistant

def create_thread():
    """
    Creates a new thread for handling conversations.
    
    Returns:
        object: The created thread.
    """
    thread = client.beta.threads.create()
    return thread
