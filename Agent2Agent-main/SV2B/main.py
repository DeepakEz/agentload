from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import json
import os
import re
import urllib.request
from pathlib import Path
from llama_cpp import Llama
from typing import Dict, List
import threading
import sqlite3
from datetime import datetime
import time
import hashlib

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

# ------------------ AGENT LEARNING SYSTEM ------------------
CONVERSATIONS_DB = "agent_conversations.db"
AGENT_WISDOM_DIR = Path("agent_wisdom")
AGENT_WISDOM_DIR.mkdir(exist_ok=True)

class ConversationDatabase:
    """Manages conversation storage and retrieval"""
    def __init__(self, db_path: str = CONVERSATIONS_DB):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Initialize database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Conversations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_id TEXT NOT NULL,
                user_message TEXT NOT NULL,
                agent_response TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                session_id TEXT NOT NULL,
                tokens_used INTEGER DEFAULT 0
            )
        ''')

        # Agent insights table (proactive messages)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS agent_insights (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_id TEXT NOT NULL,
                insight TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                delivered BOOLEAN DEFAULT FALSE
            )
        ''')

        conn.commit()
        conn.close()

    def store_conversation(self, agent_id: str, user_msg: str, agent_resp: str, session_id: str, tokens: int = 0):
        """Store a conversation exchange"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO conversations (agent_id, user_message, agent_response, timestamp, session_id, tokens_used)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (agent_id, user_msg, agent_resp, datetime.now().isoformat(), session_id, tokens))
        conn.commit()
        conn.close()

    def get_agent_conversations(self, agent_id: str, limit: int = 100):
        """Get recent conversations for an agent"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT user_message, agent_response, timestamp, tokens_used
            FROM conversations
            WHERE agent_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (agent_id, limit))
        results = cursor.fetchall()
        conn.close()
        return results

    def store_insight(self, agent_id: str, insight: str):
        """Store a proactive insight"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO agent_insights (agent_id, insight, timestamp)
            VALUES (?, ?, ?)
        ''', (agent_id, insight, datetime.now().isoformat()))
        conn.commit()
        conn.close()

    def get_pending_insights(self, agent_id: str):
        """Get undelivered insights"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id, insight, timestamp
            FROM agent_insights
            WHERE agent_id = ? AND delivered = FALSE
            ORDER BY timestamp ASC
        ''', (agent_id,))
        results = cursor.fetchall()
        conn.close()
        return results

    def mark_insight_delivered(self, insight_id: int):
        """Mark insight as delivered"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('UPDATE agent_insights SET delivered = TRUE WHERE id = ?', (insight_id,))
        conn.commit()
        conn.close()

class AgentWisdom:
    """Manages agent learning and wisdom accumulation"""
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.wisdom_file = AGENT_WISDOM_DIR / f"{agent_id}_wisdom.json"
        self.wisdom = self.load_wisdom()

    def load_wisdom(self) -> dict:
        """Load accumulated wisdom"""
        if self.wisdom_file.exists():
            with open(self.wisdom_file, 'r') as f:
                return json.load(f)
        return {
            "patterns": [],  # Learned patterns from conversations
            "preferences": {},  # User preferences
            "insights": [],  # Generated insights
            "conversation_count": 0,
            "last_learning_timestamp": None,
            "expertise_areas": []  # Topics agent has discussed
        }

    def save_wisdom(self):
        """Save wisdom to file"""
        with open(self.wisdom_file, 'w') as f:
            json.dump(self.wisdom, f, indent=2)

    def add_pattern(self, pattern: dict):
        """Add a learned pattern"""
        self.wisdom["patterns"].append({
            **pattern,
            "timestamp": datetime.now().isoformat()
        })
        # Keep only last 50 patterns
        self.wisdom["patterns"] = self.wisdom["patterns"][-50:]
        self.save_wisdom()

    def add_insight(self, insight: str):
        """Add a generated insight"""
        self.wisdom["insights"].append({
            "text": insight,
            "timestamp": datetime.now().isoformat()
        })
        self.wisdom["insights"] = self.wisdom["insights"][-20:]
        self.save_wisdom()

    def increment_conversation_count(self):
        """Increment conversation counter"""
        self.wisdom["conversation_count"] += 1
        self.save_wisdom()

    def get_wisdom_summary(self) -> str:
        """Generate a summary of accumulated wisdom"""
        if not self.wisdom["patterns"]:
            return ""

        summary_parts = ["Based on our past conversations, I've learned:"]

        # Recent patterns
        if self.wisdom["patterns"]:
            recent_patterns = self.wisdom["patterns"][-5:]
            for p in recent_patterns:
                if "description" in p:
                    summary_parts.append(f"- {p['description']}")

        # Expertise areas
        if self.wisdom["expertise_areas"]:
            areas = ", ".join(self.wisdom["expertise_areas"][-5:])
            summary_parts.append(f"\nAreas we've discussed: {areas}")

        return "\n".join(summary_parts)

class BackgroundLearner:
    """Continuous learning in background"""
    def __init__(self):
        self.db = ConversationDatabase()
        self.running = False
        self.thread = None
        self.learning_interval = 300  # Learn every 5 minutes

    def start(self):
        """Start background learning"""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._learning_loop, daemon=True)
            self.thread.start()
            print("Background learning system started")

    def stop(self):
        """Stop background learning"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)

    def _learning_loop(self):
        """Main learning loop"""
        while self.running:
            try:
                self._analyze_all_agents()
            except Exception as e:
                print(f"Background learning error: {e}")
            time.sleep(self.learning_interval)

    def _analyze_all_agents(self):
        """Analyze conversations for all agents"""
        agents = load_agents()
        for agent_id in agents.keys():
            self._analyze_agent(agent_id)

    def _analyze_agent(self, agent_id: str):
        """Analyze conversations and learn patterns for an agent"""
        conversations = self.db.get_agent_conversations(agent_id, limit=20)
        if len(conversations) < 3:
            return  # Not enough data

        wisdom = AgentWisdom(agent_id)

        # Extract topics discussed
        topics = self._extract_topics(conversations)
        if topics:
            wisdom.wisdom["expertise_areas"] = list(set(wisdom.wisdom["expertise_areas"] + topics))
            wisdom.save_wisdom()

        # Generate insight if pattern detected
        if len(conversations) >= 10 and len(conversations) % 10 == 0:
            insight = self._generate_insight(agent_id, conversations)
            if insight:
                self.db.store_insight(agent_id, insight)
                wisdom.add_insight(insight)
                print(f"Agent {agent_id} generated insight: {insight[:100]}...")

    def _extract_topics(self, conversations: List) -> List[str]:
        """Simple topic extraction from conversations"""
        topics = []
        for user_msg, agent_resp, _, _ in conversations[:5]:
            # Simple keyword extraction (can be enhanced with NLP)
            combined = (user_msg + " " + agent_resp).lower()
            keywords = ["python", "javascript", "data", "math", "code", "science", "history", "art"]
            for keyword in keywords:
                if keyword in combined and keyword not in topics:
                    topics.append(keyword)
        return topics[:5]

    def _generate_insight(self, agent_id: str, conversations: List) -> str | None:
        """Generate a proactive insight from patterns"""
        # This is a simple version - can be enhanced with actual LLM analysis
        if len(conversations) >= 10:
            return f"I've been reflecting on our {len(conversations)} conversations, and I think I can help you even better now!"
        return None

# Initialize systems
conversation_db = ConversationDatabase()
background_learner = BackgroundLearner()

# Start background learning on startup
@app.on_event("startup")
async def startup_event():
    background_learner.start()

@app.on_event("shutdown")
async def shutdown_event():
    background_learner.stop()

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

def build_context_prompt(system_instruction: str | None, conversation_history: list[Message] | None, user_prompt: str, agent_id: str | None = None) -> str:
    """Build a prompt with system instruction, wisdom, and conversation history"""
    prompt_parts = []

    # Add system instruction if provided
    if system_instruction:
        prompt_parts.append(f"System: {system_instruction}\n")

    # Add agent wisdom if available
    if agent_id:
        wisdom = AgentWisdom(agent_id)
        wisdom_summary = wisdom.get_wisdom_summary()
        if wisdom_summary:
            prompt_parts.append(f"My accumulated knowledge:\n{wisdom_summary}\n")

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

def estimate_tokens(text: str) -> int:
    """Rough token estimation (1 token ≈ 4 characters)"""
    return len(text) // 4

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
    session_id = hashlib.md5(datetime.now().isoformat().encode()).hexdigest()[:8]

    # If agent_id is provided, use that agent's system instruction
    if req.agent_id:
        system_instruction = get_agent_system_instruction(req.agent_id)
        if not system_instruction:
            return {"response": f"Error: Agent '{req.agent_id}' not found."}

        # Build context-aware prompt with agent's system instruction, wisdom, and conversation history
        context_prompt = build_context_prompt(system_instruction, req.conversation_history, user_prompt, req.agent_id)

        # Estimate tokens and warn if approaching limit
        tokens_used = estimate_tokens(context_prompt)
        if tokens_used > 1800:  # Warn at 90% of 2048 context window
            # TODO: Implement automatic summarization here
            print(f"Warning: Context approaching limit ({tokens_used} tokens)")

        response = call_model(context_prompt, max_tokens=1024)

        # Store conversation for learning
        conversation_db.store_conversation(req.agent_id, user_prompt, response, session_id, tokens_used)

        # Update agent wisdom
        wisdom = AgentWisdom(req.agent_id)
        wisdom.increment_conversation_count()

        # Check for pending insights
        insights = conversation_db.get_pending_insights(req.agent_id)

        return {
            "response": response,
            "tokens_used": tokens_used,
            "pending_insights": [{"id": i[0], "text": i[1], "timestamp": i[2]} for i in insights]
        }

    # No agent selected - respond as a general assistant
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

# Endpoint to create a new agent
@app.post("/agents")
def create_agent(request: dict):
    """Create a new agent and save to agents.json"""
    agent_name = request.get("name", "").strip()
    system_instruction = request.get("system_instruction", "").strip()

    if not agent_name:
        return {"success": False, "error": "Agent name is required"}

    if not system_instruction:
        return {"success": False, "error": "System instruction is required"}

    # Load existing agents
    agents = load_agents()

    # Check if agent already exists
    if agent_name in agents:
        return {"success": False, "error": f"Agent '{agent_name}' already exists"}

    # Add new agent
    agents[agent_name] = {"system_instruction": system_instruction}

    # Save to agents.json
    try:
        with open(AGENTS_PATH, "w") as f:
            json.dump(agents, f, indent=2)
        return {"success": True, "message": f"Agent '{agent_name}' created successfully"}
    except Exception as e:
        return {"success": False, "error": f"Failed to save agent: {str(e)}"}

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

# ==================== AGENT LEARNING ENDPOINTS ====================

@app.get("/agents/{agent_id}/wisdom")
def get_agent_wisdom(agent_id: str):
    """Get agent's accumulated wisdom and statistics"""
    wisdom = AgentWisdom(agent_id)
    conversations = conversation_db.get_agent_conversations(agent_id, limit=10)

    return {
        "agent_id": agent_id,
        "conversation_count": wisdom.wisdom["conversation_count"],
        "expertise_areas": wisdom.wisdom["expertise_areas"],
        "recent_insights": wisdom.wisdom["insights"][-5:],
        "recent_patterns": wisdom.wisdom["patterns"][-5:],
        "recent_conversations_count": len(conversations),
        "last_learning": wisdom.wisdom.get("last_learning_timestamp")
    }

@app.post("/agents/{agent_id}/insights/{insight_id}/delivered")
def mark_insight_delivered(agent_id: str, insight_id: int):
    """Mark an insight as delivered to user"""
    conversation_db.mark_insight_delivered(insight_id)
    return {"success": True}

@app.get("/agents/{agent_id}/insights")
def get_agent_insights(agent_id: str):
    """Get pending insights for an agent"""
    insights = conversation_db.get_pending_insights(agent_id)
    return {
        "insights": [
            {"id": i[0], "text": i[1], "timestamp": i[2]}
            for i in insights
        ]
    }

@app.get("/agents/{agent_id}/conversations")
def get_agent_conversation_history(agent_id: str, limit: int = 20):
    """Get conversation history for an agent"""
    conversations = conversation_db.get_agent_conversations(agent_id, limit)
    return {
        "conversations": [
            {
                "user_message": c[0],
                "agent_response": c[1],
                "timestamp": c[2],
                "tokens_used": c[3]
            }
            for c in conversations
        ]
    }
