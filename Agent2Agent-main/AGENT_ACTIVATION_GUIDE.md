# Agent Activation Feature Guide

## Overview

The agent activation feature is now **fully functional**! This critical update allows you to:

✅ **Create agents** through natural language
✅ **Activate and use agents** via dropdown selector
✅ **Maintain conversation context** across messages
✅ **Switch between agents** mid-conversation

---

## What's New

### Backend Changes (SV2B/main.py)

**1. Enhanced Request Model**
```python
class UserPromptRequest(BaseModel):
    user_prompt: str
    agent_id: str | None = None                    # NEW: Select specific agent
    conversation_history: list[Message] | None = None  # NEW: Context memory
```

**2. New API Endpoint**
- `GET /agents` - Returns list of all available agents

**3. Agent System Instruction Loading**
- Agents' system instructions are now applied to prompts
- Context-aware conversations with up to 10 previous messages

**4. Conversation Memory**
- Backend builds context from conversation history
- System instructions prepended to prompts
- Multi-turn conversations now work correctly

### Frontend Changes (SV2/agent-ui)

**1. Agent Selector Dropdown**
- Located in chat header
- Shows all available agents
- Real-time agent switching

**2. Conversation History Tracking**
- Last 10 messages sent as context
- Enables multi-turn conversations
- Agents remember previous exchanges

**3. Active Agent Display**
- Header shows current agent name
- System message when switching agents
- Clear indication of which agent is active

---

## How to Use

### Step 1: Start the Backend

```bash
cd Agent2Agent-main/SV2B
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

First run will download the default model (~4GB Mistral-7B).

### Step 2: Start the Frontend

```bash
cd Agent2Agent-main/SV2/agent-ui
npm install
npm run dev
```

Access at: http://localhost:5173

### Step 3: Create an Agent

Simply describe what you want in the chat:

**Examples:**
```
"Create a Python coding assistant that helps debug code"
"I need an agent that explains complex math concepts simply"
"Make a creative writing agent that helps with storytelling"
```

The LLM will:
1. Detect this is an agent creation request
2. Generate a suitable name (e.g., "PythonHelper")
3. Create a system instruction
4. Save to agents.json
5. Confirm creation: "Agent 'PythonHelper' created successfully!"

### Step 4: Activate the Agent

1. Open the **agent selector dropdown** (top-right of chat header)
2. Select your newly created agent
3. System message appears: "Switched to PythonHelper..."
4. Start chatting! The agent follows its system instruction

### Step 5: Test Agent Behavior

**Example with MathMaster:**
```
User: "What's the derivative of x^2?"
MathMaster: "The derivative of x² is 2x. Let me explain step-by-step:
1. Using the power rule: d/dx(x^n) = nx^(n-1)
2. Here n = 2, so we get 2x^(2-1) = 2x^1 = 2x"
```

**Switch to Default Assistant:**
```
User: "What's the derivative of x^2?"
Default: "The derivative of x² with respect to x is 2x."
```

Notice MathMaster provides more educational, step-by-step explanations per its system instruction!

---

## Pre-loaded Agents

The system comes with 2 example agents:

### 1. DataPrepBot
**System Instruction:**
> "Automate data preparation tasks in visualization tools without manual triggers. Use predefined rules and logic..."

**Best for:** Data cleaning, ETL discussions, automation workflows

### 2. MathMaster
**System Instruction:**
> "Provide step-by-step explanations of mathematical concepts and solve problems. Be patient and encouraging..."

**Best for:** Math tutoring, problem-solving, concept explanations

---

## API Reference

### POST /process_message

**Request:**
```json
{
  "user_prompt": "Explain quantum entanglement",
  "agent_id": "MathMaster",  // Optional: null for default
  "conversation_history": [  // Optional: for context
    {"text": "Hello", "type": "user"},
    {"text": "Hi! How can I help?", "type": "agent"}
  ]
}
```

**Response:**
```json
{
  "response": "Let me explain quantum entanglement step-by-step..."
}
```

### GET /agents

**Response:**
```json
{
  "agents": [
    {
      "id": "MathMaster",
      "name": "MathMaster",
      "system_instruction": "Provide step-by-step explanations..."
    },
    {
      "id": "DataPrepBot",
      "name": "DataPrepBot",
      "system_instruction": "Automate data preparation tasks..."
    }
  ]
}
```

---

## Technical Details

### Context Building

When you chat with an agent, the backend constructs:

```
System: [Agent's system instruction]

Previous conversation:
User: [Previous message 1]
Assistant: [Previous response 1]
User: [Previous message 2]
Assistant: [Previous response 2]
...

User: [Current message]
Assistant:
```

This gives the LLM:
1. **Role definition** (system instruction)
2. **Conversation memory** (last 10 messages)
3. **Current request**

### Memory Limits

- **Frontend:** Keeps last 10 messages in payload
- **Backend:** Uses all provided history
- **Model context:** 2048 tokens (configurable in main.py line 74)

---

## Conversation Examples

### Example 1: Multi-turn with MathMaster

```
[Select MathMaster from dropdown]

You: "I'm struggling with calculus"
MathMaster: "I'm here to help! Calculus can be challenging, but we'll take it step-by-step. What specific concept are you working on?"

You: "Derivatives"
MathMaster: "Great! Derivatives measure how a function changes. Let's start with the basics. Do you know the power rule?"

You: "Not really"
MathMaster: "No problem! The power rule states: if f(x) = x^n, then f'(x) = nx^(n-1). For example, if f(x) = x³, the derivative is 3x². Let's practice..."
```

Notice: MathMaster remembers context and provides step-by-step, encouraging responses.

### Example 2: Agent Switching

```
[Default Assistant selected]

You: "What's 15 * 24?"
Default: "15 × 24 = 360"

[Switch to MathMaster]

You: "What's 15 * 24?"
MathMaster: "Let me show you how to calculate 15 × 24 step-by-step:
1. Break it down: 15 × 24 = 15 × (20 + 4)
2. Distribute: (15 × 20) + (15 × 4)
3. Calculate: 300 + 60
4. Result: 360

See how we used the distributive property?"
```

---

## Troubleshooting

### Agent Not Found Error
```json
{"response": "Error: Agent 'MyAgent' not found."}
```

**Fix:** Ensure agent name matches exactly (case-sensitive). Check `/agents` endpoint.

### Conversation Not Remembering
**Issue:** Agent gives unrelated answers
**Fix:** Ensure frontend is sending `conversation_history` (check browser console)

### System Instruction Not Applied
**Issue:** Agent behaves like default assistant
**Fix:**
1. Verify `agent_id` is being sent in request
2. Check agents.json has system_instruction field
3. Restart backend to reload agents

### Model Not Loading
**Issue:** "Model not found" or download errors
**Fix:**
1. Check internet connection
2. Verify MODEL_URL is accessible
3. Clear ./models/ and retry

---

## Advanced Usage

### Custom Model with Agent System

Use a different GGUF model while keeping agent functionality:

```bash
export MODEL_URL="https://huggingface.co/TheBloke/Llama-2-13B-chat-GGUF/resolve/main/llama-2-13b-chat.Q4_K_M.gguf"
uvicorn main:app --reload
```

### Programmatic Agent Creation

Instead of natural language, directly edit `agents.json`:

```json
{
  "CodeReviewer": {
    "system_instruction": "You are a senior code reviewer. Provide constructive feedback on code quality, security, and best practices. Be thorough but kind."
  }
}
```

Restart backend to load new agent.

### Temperature Control for Agents

Edit `main.py` line 88 to adjust creativity per agent:

```python
# More creative responses
temperature=0.9,

# More focused/deterministic
temperature=0.3,
```

---

## Roadmap

This activation system enables future features:

- ✅ **Agent activation** (DONE)
- ✅ **Conversation memory** (DONE)
- 🚧 Agent editing UI
- 🚧 Agent marketplace/sharing
- 🚧 Multi-agent workflows (Agent2Agent vision)
- 🚧 RAG integration per agent
- 🚧 Fine-tuned models per agent

---

## Key Files Modified

### Backend
- `SV2B/main.py` (lines 33-40, 104-217)
  - Added agent_id and conversation_history support
  - New /agents endpoint
  - Context-aware prompt building

### Frontend
- `SV2/agent-ui/src/components/ChatWindow.jsx` (lines 11-12, 15-29, 31-46, 69-86, 145-163)
  - Agent selector dropdown
  - Conversation history tracking
  - Agent switching functionality

---

## Success Criteria

✅ Agents can be created via natural language
✅ Agents appear in dropdown selector
✅ Selecting agent activates its system instruction
✅ Conversations maintain context (10 messages)
✅ Switching agents mid-conversation works
✅ Default assistant remains available

**The core "Agent2Agent" vision is now functional!** 🎉

---

## Questions?

- Check backend logs: `uvicorn main:app --reload` output
- Check frontend console: Browser DevTools → Console
- Verify agents.json structure
- Test with simple prompts first
- Ensure both frontend and backend are running

---

**Last Updated:** January 2, 2026
**Feature Version:** 2.0 - Agent Activation Release
