from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import json
import os
import re
import urllib.request
from pathlib import Path
from llama_cpp import Llama
from typing import Dict
import threading

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
MODEL_CACHE_DIR = Path(os.getenv("MODEL_CACHE_DIR", "./models"))
MODEL_CACHE_DIR.mkdir(exist_ok=True)

AGENT_FILE = "agents.json"
MODELS_CONFIG_FILE = "models_config.json"

# Default models available
DEFAULT_MODELS = [
    {
        "id": "mistral-7b-instruct-q4",
        "name": "Mistral 7B Instruct Q4_K_M",
        "url": "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        "size": "4.4 GB",
        "description": "Balanced performance and quality"
    },
    {
        "id": "mistral-7b-instruct-q2",
        "name": "Mistral 7B Instruct Q2_K",
        "url": "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q2_K.gguf",
        "size": "3.0 GB",
        "description": "Smaller, faster, lower quality"
    },
    {
        "id": "llama2-7b-chat-q4",
        "name": "Llama 2 7B Chat Q4_K_M",
        "url": "https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf",
        "size": "4.1 GB",
        "description": "Meta's Llama 2 chat model"
    }
]

# Model management
class ModelManager:
    def __init__(self):
        self.loaded_models: Dict[str, Llama] = {}
        self.active_model_id: str | None = None
        self.download_progress: Dict[str, dict] = {}
        self.lock = threading.Lock()

    def get_active_model(self) -> Llama | None:
        if self.active_model_id and self.active_model_id in self.loaded_models:
            return self.loaded_models[self.active_model_id]
        return None

    def load_model_by_id(self, model_id: str) -> bool:
        """Load a model by its ID"""
        with self.lock:
            # Check if already loaded
            if model_id in self.loaded_models:
                self.active_model_id = model_id
                return True

            # Find model config
            model_info = self.get_model_info(model_id)
            if not model_info:
                return False

            # Get model path
            model_filename = model_info["url"].split("/")[-1]
            model_path = MODEL_CACHE_DIR / model_filename

            if not model_path.exists():
                return False  # Model not downloaded

            # Load model
            print(f"Loading model {model_id} from {model_path}...")
            try:
                llm = Llama(
                    model_path=str(model_path),
                    n_ctx=2048,
                    n_threads=4,
                    n_gpu_layers=0,
                )
                self.loaded_models[model_id] = llm
                self.active_model_id = model_id
                print(f"Model {model_id} loaded successfully!")
                return True
            except Exception as e:
                print(f"Failed to load model {model_id}: {e}")
                return False

    def get_model_info(self, model_id: str) -> dict | None:
        """Get model configuration by ID"""
        for model in DEFAULT_MODELS:
            if model["id"] == model_id:
                return model
        # Check custom models
        custom_models = self.load_custom_models()
        for model in custom_models:
            if model["id"] == model_id:
                return model
        return None

    def load_custom_models(self) -> list:
        """Load custom models from config file"""
        if os.path.exists(MODELS_CONFIG_FILE):
            with open(MODELS_CONFIG_FILE, "r") as f:
                try:
                    return json.load(f)
                except:
                    return []
        return []

    def save_custom_model(self, model_info: dict):
        """Save a custom model to config"""
        custom_models = self.load_custom_models()
        # Check if already exists
        for i, m in enumerate(custom_models):
            if m["id"] == model_info["id"]:
                custom_models[i] = model_info
                break
        else:
            custom_models.append(model_info)

        with open(MODELS_CONFIG_FILE, "w") as f:
            json.dump(custom_models, f, indent=2)

# Global model manager
model_manager = ModelManager()

# ------------------ REQUEST MODEL ------------------
class Message(BaseModel):
    text: str
    type: str  # "user" or "agent"

class UserPromptRequest(BaseModel):
    user_prompt: str
    agent_id: str | None = None  # Optional: specific agent to use
    conversation_history: list[Message] | None = None  # Optional: previous messages for context

# ------------------ UTILS ------------------
def download_model_with_progress(model_id: str, url: str, destination: Path):
    """Download GGUF model from URL with progress tracking"""
    print(f"Downloading model {model_id} from {url}...")

    def report_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100, (downloaded / total_size) * 100)
            # Update progress in model_manager
            model_manager.download_progress[model_id] = {
                "status": "downloading",
                "progress": percent,
                "downloaded": downloaded,
                "total": total_size
            }
            print(f"\rProgress: {percent:.1f}%", end="", flush=True)

    try:
        model_manager.download_progress[model_id] = {
            "status": "downloading",
            "progress": 0,
            "downloaded": 0,
            "total": 0
        }
        urllib.request.urlretrieve(url, destination, reporthook=report_progress)
        model_manager.download_progress[model_id] = {
            "status": "completed",
            "progress": 100
        }
        print(f"\nModel {model_id} downloaded successfully to {destination}")
    except Exception as e:
        model_manager.download_progress[model_id] = {
            "status": "failed",
            "error": str(e)
        }
        print(f"\nFailed to download model {model_id}: {e}")
        raise

def call_model(prompt: str, max_tokens: int = 512) -> str:
    """Call the active GGUF model with a prompt"""
    try:
        llm = model_manager.get_active_model()
        if not llm:
            # Try to load default model if no model is active
            if not model_manager.load_model_by_id("mistral-7b-instruct-q4"):
                return "[Error: No model loaded. Please download and select a model first.]"
            llm = model_manager.get_active_model()

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

def extract_json_from_text(text: str) -> dict | None:
    """Extract JSON object from text that may contain additional content"""
    # Try direct JSON parsing first
    try:
        return json.loads(text.strip())
    except:
        pass

    # Find JSON object using regex (match content between { and })
    json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except:
            pass

    # Try to find JSON between common delimiters
    for pattern in [r'```json\s*(\{.*?\})\s*```', r'```\s*(\{.*?\})\s*```']:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except:
                pass

    return None

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

    # Extract JSON from response (handles extra text around JSON)
    llm_data = extract_json_from_text(llm_response)
    if not llm_data:
        # If JSON extraction fails, log the response for debugging
        print(f"Failed to parse LLM response: {llm_response}")
        return {"response": "Error parsing LLM response. The model may need adjustment."}

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

# ==================== MODEL MANAGEMENT ENDPOINTS ====================

@app.get("/models")
def list_models():
    """List all available models (default + custom)"""
    all_models = DEFAULT_MODELS + model_manager.load_custom_models()

    # Add download status and loaded status for each model
    for model in all_models:
        model_filename = model["url"].split("/")[-1]
        model_path = MODEL_CACHE_DIR / model_filename

        model["downloaded"] = model_path.exists()
        model["loaded"] = model["id"] == model_manager.active_model_id
        model["file_size_mb"] = round(model_path.stat().st_size / (1024 * 1024), 2) if model_path.exists() else None

    return {"models": all_models, "active_model_id": model_manager.active_model_id}

@app.post("/models/download")
async def download_model_endpoint(background_tasks: BackgroundTasks, model_id: str):
    """Download a model in the background"""
    model_info = model_manager.get_model_info(model_id)
    if not model_info:
        return {"success": False, "error": "Model not found"}

    model_filename = model_info["url"].split("/")[-1]
    model_path = MODEL_CACHE_DIR / model_filename

    if model_path.exists():
        return {"success": False, "error": "Model already downloaded"}

    # Start download in background
    def download_task():
        try:
            download_model_with_progress(model_id, model_info["url"], model_path)
        except Exception as e:
            print(f"Download failed: {e}")

    background_tasks.add_task(download_task)

    return {
        "success": True,
        "message": f"Downloading {model_info['name']}...",
        "model_id": model_id
    }

@app.get("/models/download-progress/{model_id}")
def get_download_progress(model_id: str):
    """Get download progress for a specific model"""
    progress = model_manager.download_progress.get(model_id, {"status": "not_started"})
    return progress

@app.post("/models/select")
def select_model(model_id: str):
    """Load and activate a specific model"""
    model_info = model_manager.get_model_info(model_id)
    if not model_info:
        return {"success": False, "error": "Model not found"}

    model_filename = model_info["url"].split("/")[-1]
    model_path = MODEL_CACHE_DIR / model_filename

    if not model_path.exists():
        return {"success": False, "error": "Model not downloaded. Please download it first."}

    success = model_manager.load_model_by_id(model_id)
    if success:
        return {
            "success": True,
            "message": f"Model {model_info['name']} loaded successfully",
            "active_model_id": model_manager.active_model_id
        }
    else:
        return {"success": False, "error": "Failed to load model"}

@app.post("/models/add-custom")
def add_custom_model(model_info: dict):
    """Add a custom model URL"""
    required_fields = ["id", "name", "url", "size", "description"]
    if not all(field in model_info for field in required_fields):
        return {"success": False, "error": f"Missing required fields: {required_fields}"}

    model_manager.save_custom_model(model_info)
    return {"success": True, "message": "Custom model added successfully"}

@app.delete("/models/{model_id}")
def delete_model(model_id: str):
    """Delete a downloaded model file"""
    model_info = model_manager.get_model_info(model_id)
    if not model_info:
        return {"success": False, "error": "Model not found"}

    model_filename = model_info["url"].split("/")[-1]
    model_path = MODEL_CACHE_DIR / model_filename

    if not model_path.exists():
        return {"success": False, "error": "Model file not found"}

    # Unload if currently loaded
    if model_id == model_manager.active_model_id:
        if model_id in model_manager.loaded_models:
            del model_manager.loaded_models[model_id]
        model_manager.active_model_id = None

    # Delete file
    try:
        model_path.unlink()
        return {"success": True, "message": "Model deleted successfully"}
    except Exception as e:
        return {"success": False, "error": str(e)}
