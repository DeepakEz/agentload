# Agent2Agent - Self-Learning AI Agent System

A revolutionary local AI agent platform that learns from every conversation, thinks in the background, and becomes wiser over time. Built with FastAPI, React, and direct GGUF model loading.

## 🌟 Key Features

### 1. **Self-Learning Agents**
- Agents learn from every conversation and store knowledge permanently
- Background AI analysis runs every 5 minutes
- Accumulates wisdom in topics, patterns, and user preferences
- Gets smarter the more you chat

### 2. **AI-Powered Background Thinking**
- Real LLM analysis of conversation patterns
- Generates creative insights and novel suggestions
- Proactively messages you with discoveries
- Continues thinking even when you're not chatting

### 3. **Conversation Persistence**
- All conversations saved to SQLite database
- Conversations automatically load when you return
- Agent remembers context across browser refreshes
- Separate conversation history per agent

### 4. **AI-Generated Agent Creation**
- Describe what you want in plain English
- AI generates agent name and system instructions automatically
- Review and edit before creating
- No manual configuration needed

### 5. **Direct GGUF Model Loading**
- No Ollama required - models run directly in Python
- Download models from any URL (HuggingFace, etc.)
- Built-in model management UI
- Automatic caching and progress tracking

### 6. **Intelligent Context Management**
- Real-time token usage monitoring
- Warns when approaching context limit (2048 tokens)
- Extracts patterns for conversation continuity
- Smart context summarization

### 7. **Vector Database Semantic Search**
- ChromaDB integration for semantic conversation search
- Find similar past conversations by meaning, not keywords
- Automatic embedding generation for all conversations
- Query past interactions with natural language

### 8. **Tool Registry & Capabilities**
- Safe, sandboxed tool execution framework
- Extensible tool system for agent capabilities
- Built-in tools: calculator, text analyzer, timestamp
- Foundation for dynamic tool creation

### 9. **Wisdom Pruning System**
- Automatic memory optimization prevents bloat
- Consolidates expertise areas intelligently
- Removes duplicates and low-quality patterns
- Runs automatically every 10 conversations

### 10. **Master Architect Meta-Agent**
- Meta-agent that optimizes other agents
- Deduplicates insights across all agents
- Consolidates similar expertise areas
- Health monitoring for agent performance

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Node.js 16+
- 8GB RAM minimum (16GB recommended for larger models)

### Installation

**1. Clone the repository:**
```bash
git clone https://github.com/yourusername/agentload.git
cd agentload/Agent2Agent-main
```

**2. Install backend dependencies:**
```bash
cd SV2B
pip install -r requirements.txt
```

**Note:** ChromaDB and sentence-transformers are included for vector search. If you encounter installation issues, the system will work without them (vector search will be disabled).

**3. Install frontend dependencies:**
```bash
cd ../SV2/agent-ui
npm install
```

### Running the Application

**Terminal 1 - Backend:**
```bash
cd Agent2Agent-main/SV2B
python -m uvicorn main:app --reload
```

**Terminal 2 - Frontend:**
```bash
cd Agent2Agent-main/SV2/agent-ui
npm run dev
```

**3. Open your browser:**
Navigate to `http://localhost:5173`

## 📖 How to Use

### First Time Setup

**1. Download a Model:**
- Click the **🤖 Models** button in the sidebar
- Click **Download** on any model (Mistral 7B Q4 recommended)
- Wait for download to complete
- Click **Select** to activate the model

**2. Create Your First Agent:**
- Click **Agents** in the sidebar
- Click **+ Add New Agent**
- Describe what you want: *"I want an agent that helps with Python coding"*
- Click **✨ Generate Agent**
- Review the generated name and instructions
- Click **Create Agent**

**3. Start Chatting:**
- Select your agent from the dropdown
- Start a conversation
- The agent learns from every message

### Using the Learning System

**Automatic Learning:**
- Every conversation is automatically saved
- Background analysis runs every 5 minutes
- Agent extracts topics and patterns
- Generates insights when discoveries are made

**Viewing Agent Wisdom:**
- Wisdom stats appear in the chat header
- Shows conversation count and expertise areas
- Example: *"📚 25 convos | python, data, algorithms"*

**Receiving Insights:**
- Agents proactively message you with discoveries
- Green messages with 💡 icon
- Appear automatically when insight is generated
- Based on deep analysis of your conversations

**Conversation Persistence:**
- All conversations saved automatically
- Refresh the page - conversations remain
- Switch agents - each has separate history
- Last 50 conversations loaded per agent

## 🛠️ Technical Architecture

### Backend (FastAPI)
```
SV2B/
├── main.py                  # Core server and API endpoints
├── agents.json              # Agent definitions
├── models/                  # Downloaded GGUF models
├── agent_wisdom/            # Per-agent learning data (JSON)
├── agent_conversations.db   # SQLite conversation database
└── chroma_db/               # Vector embeddings storage (optional)
```

### Frontend (React + Vite)
```
SV2/agent-ui/
├── src/
│   ├── App.jsx             # Main app component
│   └── components/
│       ├── Sidebar.jsx     # Navigation + agent/model management
│       ├── ChatWindow.jsx  # Chat interface + history loading
│       ├── ChatMessage.jsx # Message rendering
│       └── ModelManager.jsx # Model download UI
└── public/
```

### Key Technologies
- **Backend**: FastAPI, llama-cpp-python, SQLite, ChromaDB
- **Frontend**: React 19, Vite
- **AI**: Direct GGUF model inference, sentence-transformers
- **Storage**: SQLite (conversations) + JSON (wisdom) + ChromaDB (vectors)

## 📡 API Endpoints

### Agent Management
- `GET /agents` - List all agents
- `POST /agents` - Create new agent
- `POST /agents/generate` - AI-generate agent details
- `GET /agents/{agent_id}/wisdom` - Get agent learning stats
- `GET /agents/{agent_id}/conversations` - Get conversation history
- `GET /agents/{agent_id}/insights` - Get pending proactive insights
- `GET /agents/{agent_id}/vector-search` - Semantic search through conversations
- `POST /agents/{agent_id}/optimize` - Run Master Architect optimization
- `POST /agents/optimize-all` - Optimize all agents
- `GET /agents/{agent_id}/health` - Get agent health metrics

### Tool Management
- `GET /tools` - List all available tools
- `POST /tools/{tool_name}/execute` - Execute a specific tool

### System
- `GET /system/status` - Overall system status and capabilities

### Model Management
- `GET /models` - List available models
- `POST /models/download` - Download model in background
- `POST /models/select` - Load and activate model
- `DELETE /models/{model_id}` - Delete model file
- `GET /models/download-progress/{model_id}` - Check download status

### Chat
- `POST /process_message` - Send message and get response

## 🧠 Learning System Details

### Background Learning Process
1. **Every 5 minutes**, the BackgroundLearner analyzes all agents
2. **Topic Extraction**: LLM analyzes conversations to extract themes
3. **Pattern Detection**: Identifies recurring user interests and challenges
4. **Insight Generation**: Creates creative observations and suggestions
5. **Proactive Messaging**: Sends insights to user when discovered

### Database Schema

**conversations table:**
```sql
- id (INTEGER PRIMARY KEY)
- agent_id (TEXT)
- user_message (TEXT)
- agent_response (TEXT)
- session_id (TEXT)
- timestamp (TEXT)
- tokens_used (INTEGER)
```

**agent_insights table:**
```sql
- id (INTEGER PRIMARY KEY)
- agent_id (TEXT)
- insight_text (TEXT)
- timestamp (TEXT)
- delivered (INTEGER)
```

### Wisdom Structure (JSON per agent)
```json
{
  "patterns": [],
  "preferences": {},
  "insights": [],
  "conversation_count": 0,
  "last_learning_timestamp": null,
  "expertise_areas": ["python", "algorithms", "data-structures"]
}
```

## 📊 Model Options

### Included Default Models
1. **Mistral 7B Instruct Q4_K_M** (4.4 GB)
   - Best balance of quality and performance
   - Recommended for most users

2. **Mistral 7B Instruct Q2_K** (3.0 GB)
   - Faster inference, lower quality
   - Good for limited RAM

3. **Llama 2 7B Chat Q4_K_M** (4.1 GB)
   - Alternative model option
   - Meta's Llama 2

### Adding Custom Models
Use the "Add Custom Model" button in the Models UI to add any GGUF model URL.

## 🚀 Advanced Features

### Vector Semantic Search

Search past conversations by meaning, not just keywords:

```python
# API Usage
GET /agents/{agent_id}/vector-search?query=machine%20learning%20advice&limit=5
```

**Example:** Instead of searching for exact words, ask "what did we discuss about optimization?" and it will find related conversations about performance, speed, efficiency, etc.

**Note:** Requires ChromaDB installation. See installation section below.

### Tool Execution

Agents can use tools for enhanced capabilities:

**Available Tools:**
- `calculator` - Safe mathematical calculations
- `text_analyzer` - Word/character/line counts
- `timestamp` - Current time in various formats

```python
# List tools
GET /tools

# Execute tool
POST /tools/calculator/execute
{
  "args": ["2 + 2 * 3"]
}
```

### Master Architect Optimization

Run meta-agent optimization to improve agent performance:

```python
# Optimize single agent
POST /agents/{agent_id}/optimize

# Optimize all agents
POST /agents/optimize-all

# Check agent health
GET /agents/{agent_id}/health
```

**What it does:**
- Deduplicates insights (removes redundant learnings)
- Consolidates expertise areas (merges "ml", "ai", "neural-networks" → "machine-learning")
- Removes low-quality patterns
- Reports optimization statistics

**When to use:** Run monthly or when agents accumulate 100+ conversations.

### Wisdom Pruning

Automatic memory optimization runs every 10 conversations:
- Limits expertise areas to 15 most relevant
- Removes duplicates
- Prevents unbounded memory growth

No configuration needed - works automatically.

## 🔧 Configuration

### Environment Variables
```bash
# Optional - defaults shown
MODEL_CACHE_DIR=./models        # Where to store downloaded models
```

### Adjusting Learning Frequency
In `main.py`, modify the BackgroundLearner:
```python
self.learning_interval = 300  # Seconds (default: 5 minutes)
```

### Context Window Size
Default: 2048 tokens (warns at 1800)
Modify in `main.py`:
```python
if tokens_used > 1800:  # 90% warning threshold
```

## 🎯 Use Cases

### 1. **Personal Coding Assistant**
- Create specialized agents for different languages
- Agent learns your coding style and preferences
- Proactive suggestions based on past challenges

### 2. **Learning Companion**
- Agent tracks topics you're studying
- Identifies knowledge gaps and connections
- Suggests next learning steps

### 3. **Research Assistant**
- Remembers context across multiple sessions
- Finds patterns in research questions
- Generates novel research directions

### 4. **Creative Writing Partner**
- Learns your writing style
- Suggests plot developments based on patterns
- Maintains character consistency

## 🐛 Troubleshooting

### "No model loaded" Error
**Solution**: Download and select a model from 🤖 Models menu first.

### Agent Generation Fails
**Solution**: Ensure a model is loaded and selected. Check backend terminal for error logs.

### Conversations Don't Load
**Solution**: Check that `agent_conversations.db` exists in `SV2B/` directory. Permissions may need adjustment.

### Backend Crashes on Model Load
**Solution**: Model too large for RAM. Try Q2 quantization or smaller model.

### Port Already in Use
**Backend (8000)**:
```bash
# Windows
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Linux/Mac
lsof -ti:8000 | xargs kill -9
```

**Frontend (5173)**: Vite will suggest alternate port automatically.

## 📝 Development

### Adding New Agent Types
Edit `agents.json` directly or use the UI:
```json
{
  "AgentName": {
    "system_instruction": "Your instructions here..."
  }
}
```

### Modifying Learning Behavior
Key function in `main.py`:
- `_extract_topics_with_ai()` - Topic extraction logic
- `_generate_intelligent_insight()` - Insight generation prompt
- `_analyze_agent()` - Main analysis flow

### Frontend Customization
- Theme: Edit `themes` in `App.jsx`
- Styling: Inline styles in each component
- Polling frequency: Modify `setInterval` in ChatWindow.jsx

## 🚨 Important Notes

### Privacy
- All data stored locally on your machine
- No external API calls except model downloads
- Conversations never leave your computer

### Performance
- First message may be slow (model loading)
- Subsequent messages are faster
- Background learning uses minimal resources

### Storage
- Models: 3-5 GB per model
- Database: ~1 MB per 1000 conversations
- Wisdom files: <100 KB per agent

## 📚 Documentation

- [GGUF Model Guide](Agent2Agent-main/README_GGUF.md) - Direct model loading
- [Model Management](Agent2Agent-main/MODEL_MANAGEMENT_GUIDE.md) - UI usage
- [Agent Learning System](Agent2Agent-main/AGENT_LEARNING_GUIDE.md) - Deep dive
- [Agent Activation](Agent2Agent-main/AGENT_ACTIVATION_GUIDE.md) - How agents work

## 🤝 Contributing

Contributions welcome! This is a growing project focused on local AI agents that truly learn.

## 📄 License

MIT License - See LICENSE file for details

## 🎉 Acknowledgments

- Built on [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)
- Models from [TheBloke on HuggingFace](https://huggingface.co/TheBloke)
- FastAPI and React communities

---

**Made with 🧠 by the Agent2Agent Team**

*Building the future of self-learning AI agents, locally.*
