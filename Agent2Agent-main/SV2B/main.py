from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import json
import os
import urllib.request
from pathlib import Path
from llama_cpp import Llama

# ------------------ INIT ------------------
app = FastAPI(title="Local GGUF Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------ MODEL CONFIGURATION ------------------
# You can change this to any GGUF model URL
MODEL_URL = os.getenv("MODEL_URL", "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf")
MODEL_CACHE_DIR = Path(os.getenv("MODEL_CACHE_DIR", "./models"))
MODEL_CACHE_DIR.mkdir(exist_ok=True)

AGENT_FILE = "agents.json"

# Global model instance
llm_model = None

# ------------------ REQUEST MODEL ------------------
class Message(BaseModel):
    text: str
    type: str  # "user" or "agent"

class UserPromptRequest(BaseModel):
    user_prompt: str
    agent_id: str | None = None  # Optional: specific agent to use
    conversation_history: list[Message] | None = None  # Optional: previous messages for context

# ------------------ UTILS ------------------
def download_model(url: str, destination: Path) -> Path:
    """Download GGUF model from URL to local cache"""
    print(f"Downloading model from {url}...")
    print(f"This may take a while depending on model size...")

    def report_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100, (downloaded / total_size) * 100)
            print(f"\rProgress: {percent:.1f}%", end="", flush=True)

    urllib.request.urlretrieve(url, destination, reporthook=report_progress)
    print(f"\nModel downloaded successfully to {destination}")
    return destination

def load_model() -> Llama:
    """Load GGUF model from URL or local cache"""
    global llm_model

    if llm_model is not None:
        return llm_model

    # Get model filename from URL
    model_filename = MODEL_URL.split("/")[-1]
    model_path = MODEL_CACHE_DIR / model_filename

    # Download if not cached
    if not model_path.exists():
        print(f"Model not found in cache at {model_path}")
        model_path = download_model(MODEL_URL, model_path)
    else:
        print(f"Using cached model at {model_path}")

    # Load model with llama-cpp-python
    print("Loading model into memory...")
    llm_model = Llama(
        model_path=str(model_path),
        n_ctx=2048,  # Context window
        n_threads=4,  # Number of CPU threads
        n_gpu_layers=0,  # Set to >0 if you have GPU support
    )
    print("Model loaded successfully!")
    return llm_model

def call_model(prompt: str, max_tokens: int = 512) -> str:
    """Call the GGUF model with a prompt"""
    try:
        llm = load_model()
        response = llm(
            prompt,
            max_tokens=max_tokens,
            temperature=0.7,
            top_p=0.9,
            echo=False,
            stop=["</s>", "User:", "\n\n\n"]
        )
        return response["choices"][0]["text"].strip()
    except Exception as e:
        print(f"Model call failed: {e}")
        return "[Error generating response]"

def load_agents() -> dict:
    """Load all agents from agents.json"""
    if os.path.exists(AGENT_FILE):
        with open(AGENT_FILE, "r") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {}
    return {}

def get_agent_system_instruction(agent_id: str) -> str | None:
    """Get system instruction for a specific agent"""
    agents = load_agents()
    agent = agents.get(agent_id)
    if agent:
        return agent.get("system_instruction")
    return None

def save_agent(agent_name: str, system_instruction: str):
    """Save a new agent to agents.json"""
    agents = load_agents()
    agents[agent_name] = {"system_instruction": system_instruction}
    with open(AGENT_FILE, "w") as f:
        json.dump(agents, f, indent=2)

def build_context_prompt(system_instruction: str | None, conversation_history: list[Message] | None, user_prompt: str) -> str:
    """Build a prompt with system instruction and conversation history"""
    prompt_parts = []

    # Add system instruction if provided
    if system_instruction:
        prompt_parts.append(f"System: {system_instruction}\n")

    # Add conversation history (last 10 messages for context)
    if conversation_history:
        prompt_parts.append("Previous conversation:")
        for msg in conversation_history[-10:]:  # Keep last 10 messages
            role = "User" if msg.type == "user" else "Assistant"
            prompt_parts.append(f"{role}: {msg.text}")
        prompt_parts.append("")  # Empty line separator

    # Add current user prompt
    prompt_parts.append(f"User: {user_prompt}")
    prompt_parts.append("Assistant:")

    return "\n".join(prompt_parts)

@app.post("/process_message")
def process_prompt(req: UserPromptRequest):
    user_prompt = req.user_prompt.strip()

    # If agent_id is provided, use that agent's system instruction
    if req.agent_id:
        system_instruction = get_agent_system_instruction(req.agent_id)
        if not system_instruction:
            return {"response": f"Error: Agent '{req.agent_id}' not found."}

        # Build context-aware prompt with agent's system instruction and conversation history
        context_prompt = build_context_prompt(system_instruction, req.conversation_history, user_prompt)
        response = call_model(context_prompt, max_tokens=1024)
        return {"response": response}

    # No agent selected - check if this is an agent creation request
    # Step 1: Ask LLM to detect if this is an agent creation request
    llm_instruction = f"""
You are an AI assistant that helps create other AI agents for local use.

User input: "{user_prompt}"

Your tasks:
1. Determine if this is a request to create a new agent.
2. If yes:
   - Give the agent a concise name (agent_name).
   - Identify its role/persona (agent_role).
   - Write a system instruction (system_instruction) that will guide this agent's behavior when deployed locally.
     The system instruction should be concise, actionable, and suitable for a local AI to follow in responding to user prompts.
3. If not an agent creation request, return is_agent_creation=false.

Output strictly JSON with these keys:
{{
  "agent_name": <name or null>,
  "agent_role": <role or null>,
  "system_instruction": <instruction or null>,
  "is_agent_creation": <true/false>
}}
"""

    # Step 2: Call your LLM with GGUF model
    llm_response = call_model(llm_instruction, max_tokens=512)  # returns JSON

    try:
        llm_data = json.loads(llm_response)
    except Exception:
        return {"response": "Error parsing LLM response."}

    # Step 3: If LLM confirms agent creation, save agent
    if llm_data.get("is_agent_creation"):
        agent_name = llm_data.get("agent_name") or "CustomAgent"
        system_instruction = llm_data.get("system_instruction") or ""
        save_agent(agent_name, system_instruction)
        return {"response": f"Agent '{agent_name}' created successfully!"}

    # Step 4: Otherwise, treat it as a regular prompt with conversation history
    context_prompt = build_context_prompt(None, req.conversation_history, user_prompt)
    regular_response = call_model(context_prompt, max_tokens=1024)
    return {"response": regular_response}

# New endpoint to list all available agents
@app.get("/agents")
def list_agents():
    """Return list of all available agents"""
    agents = load_agents()
    return {"agents": [{"id": name, "name": name, "system_instruction": data.get("system_instruction")}
                       for name, data in agents.items()]}
