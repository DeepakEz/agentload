from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import subprocess
import json
import os

# ------------------ INIT ------------------
app = FastAPI(title="Local Ollama Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_NAME = "llama3"
OLLAMA_CMD = ["ollama", "run", MODEL_NAME]
AGENT_FILE = "agents.json"

# ------------------ REQUEST MODEL ------------------
class UserPromptRequest(BaseModel):
    user_prompt: str

# ------------------ UTILS ------------------
def call_ollama(prompt: str) -> str:
    try:
        result = subprocess.run(
            OLLAMA_CMD,
            input=prompt,
            text=True,
            capture_output=True,
            check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print("Ollama call failed:", e)
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


    # Step 2: Call your LLM (Ollama or any model)
    llm_response = call_ollama(llm_instruction)  # returns JSON

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
    regular_response = call_ollama(user_prompt)
    return {"response": regular_response}
