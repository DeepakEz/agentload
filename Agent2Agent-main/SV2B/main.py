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
class UserPromptRequest(BaseModel):
    user_prompt: str

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

def save_agent(agent_name: str, system_instruction: str):
    agents = {}
    if os.path.exists(AGENT_FILE):
        with open(AGENT_FILE, "r") as f:
            try:
                agents = json.load(f)
            except json.JSONDecodeError:
                agents = {}
    agents[agent_name] = {"system_instruction": system_instruction}
    with open(AGENT_FILE, "w") as f:
        json.dump(agents, f, indent=2)

@app.post("/process_message")
def process_prompt(req: UserPromptRequest):
    user_prompt = req.user_prompt.strip()

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

    # Step 4: Otherwise, treat it as a regular prompt
    regular_response = call_model(user_prompt, max_tokens=1024)
    return {"response": regular_response}
