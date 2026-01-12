# Agent Learning System Guide

## 🧠 Overview

The Agent2Agent platform now features a **revolutionary self-learning system** where agents continuously improve through conversation, analyze their own responses, and proactively engage users with novel insights.

### Key Features
- ✅ **Continuous Learning**: Agents learn from every conversation
- ✅ **Pattern Recognition**: Extract themes and user preferences
- ✅ **Wisdom Accumulation**: Knowledge persists across sessions
- ✅ **Context Awareness**: Monitors token usage, warns near limits
- ✅ **Proactive Engagement**: Agents initiate messages with insights
- ✅ **Background Processing**: Learning continues even when idle
- ✅ **Evolution Over Time**: The more you chat, the wiser they become

---

## 🎯 How It Works

### 1. **Conversation Storage**

Every message exchange is stored in SQLite database:

```
agent_conversations.db
├── conversations table
│   ├── user_message
│   ├── agent_response
│   ├── timestamp
│   ├── session_id
│   └── tokens_used
└── agent_insights table
    ├── insight text
    ├── timestamp
    └── delivered status
```

**What's Stored:**
- All user prompts and agent responses
- Timestamps for temporal analysis
- Token usage for context management
- Session IDs for conversation grouping

### 2. **Wisdom Accumulation**

Each agent has a personal "wisdom file":

```json
// agent_wisdom/MathMaster_wisdom.json
{
  "patterns": [
    {
      "description": "User prefers step-by-step explanations",
      "timestamp": "2026-01-12T10:30:00"
    }
  ],
  "preferences": {},
  "insights": [
    {
      "text": "I've noticed you struggle with calculus derivatives",
      "timestamp": "2026-01-12T11:00:00"
    }
  ],
  "conversation_count": 25,
  "last_learning_timestamp": "2026-01-12T11:05:00",
  "expertise_areas": ["calculus", "algebra", "geometry"]
}
```

**What's Learned:**
- Communication patterns
- User preferences and habits
- Topics discussed (expertise areas)
- Generated insights
- Conversation statistics

### 3. **Background Learning**

A dedicated thread runs every 5 minutes:

```
Background Learner Thread
↓
Analyze all agents
↓
Extract topics from recent conversations
↓
Detect patterns (every 10 conversations)
↓
Generate insights
↓
Store in database
↓
Update wisdom files
```

**What It Does:**
- Analyzes last 20 conversations per agent
- Extracts keywords (python, math, code, etc.)
- Generates insights at conversation milestones (10, 20, 30...)
- Runs continuously, even when user isn't chatting

### 4. **Proactive Engagement**

Agents can initiate messages:

```
Agent analyzes conversations
   ↓
Detects a pattern or solves a problem
   ↓
Generates an insight
   ↓
Stores in agent_insights table
   ↓
Frontend polls for insights every 30s
   ↓
Displays as special message: 💡
   ↓
Marks as delivered
```

**Example:**
```
Agent: 💡 I've been reflecting on our 15 conversations, and I think I can help you even better now!
```

### 5. **Context Management**

Monitors token usage to prevent context overflow:

```
Build prompt with:
- System instruction
- Agent wisdom summary
- Last 10 messages
- Current prompt
   ↓
Estimate tokens (text length / 4)
   ↓
Display: "Context: 1234 / 2048 tokens"
   ↓
If > 1800: Show "⚠️ Near limit"
   ↓
(Future: Auto-summarize when approaching limit)
```

---

## 📊 UI Indicators

### Header Display

When chatting with an agent, the header shows:

```
╔══════════════════════════════════════════════╗
║ Chatting with: MathMaster                    ║
║ 📚 25 convos | calculus, algebra, geometry   ║
║ Context: 1234 / 2048 tokens                  ║
╚══════════════════════════════════════════════╝
```

**Elements:**
- **Agent name**: Currently active agent
- **📚 Badge**: Conversation count + expertise areas
- **Token counter**: Context window usage
- **⚠️ Warning**: Appears at 90% context (1800+ tokens)

### Insight Messages

Proactive messages appear centered with special styling:

```
┌──────────────────────────────────────────────┐
│  💡 I've been reflecting on our              │
│  conversations and discovered a pattern...   │
└──────────────────────────────────────────────┘
```

**Visual Features:**
- Dark green background (#1a472a)
- Light green text (#90EE90)
- Green border
- Centered alignment
- Italic font
- Fade-in animation

---

## 🔧 API Reference

### Get Agent Wisdom

```bash
GET /agents/{agent_id}/wisdom
```

**Response:**
```json
{
  "agent_id": "MathMaster",
  "conversation_count": 25,
  "expertise_areas": ["calculus", "algebra", "geometry"],
  "recent_insights": [
    {
      "text": "I've noticed patterns in your questions",
      "timestamp": "2026-01-12T10:00:00"
    }
  ],
  "recent_patterns": [...],
  "recent_conversations_count": 10,
  "last_learning": "2026-01-12T11:00:00"
}
```

### Get Pending Insights

```bash
GET /agents/{agent_id}/insights
```

**Response:**
```json
{
  "insights": [
    {
      "id": 1,
      "text": "I think I found a better way to explain derivatives!",
      "timestamp": "2026-01-12T10:30:00"
    }
  ]
}
```

### Mark Insight Delivered

```bash
POST /agents/{agent_id}/insights/{insight_id}/delivered
```

**Response:**
```json
{
  "success": true
}
```

### Get Conversation History

```bash
GET /agents/{agent_id}/conversations?limit=20
```

**Response:**
```json
{
  "conversations": [
    {
      "user_message": "What's the derivative of x²?",
      "agent_response": "The derivative is 2x...",
      "timestamp": "2026-01-12T10:00:00",
      "tokens_used": 234
    }
  ]
}
```

### Process Message (Enhanced)

```bash
POST /process_message
```

**Request:**
```json
{
  "user_prompt": "Explain integration",
  "agent_id": "MathMaster",
  "conversation_history": [...]
}
```

**Response:**
```json
{
  "response": "Integration is the reverse of differentiation...",
  "tokens_used": 456,
  "pending_insights": [
    {
      "id": 2,
      "text": "I've learned 3 new calculus topics!",
      "timestamp": "2026-01-12T10:35:00"
    }
  ]
}
```

---

## 🎓 Learning Examples

### Example 1: Pattern Detection

**Conversations 1-10** (User asks about Python):
```
User: "How do I use list comprehensions?"
Agent: "List comprehensions allow you to..."
[...more Python questions...]
```

**After 10 conversations:**
```
Background Learner analyzes:
- Keywords found: "python" (8 times), "code" (5 times)
- Expertise area added: "python"
- Wisdom file updated
```

**In Header:**
```
📚 10 convos | python
```

### Example 2: Proactive Insight

**After 10 conversations:**
```
Background Learner generates:
"I've been reflecting on our 10 conversations,
and I think I can help you even better now!"
```

**User sees:**
```
💡 I've been reflecting on our 10 conversations,
and I think I can help you even better now!
```

### Example 3: Context Warning

**Long conversation:**
```
Context: 1845 / 2048 tokens ⚠️ Near limit
```

**What happens:**
- Warning appears in header
- Backend logs: "Warning: Context approaching limit"
- (Future: Auto-summarization triggers)

---

## 🚀 How to Experience Learning

### Step 1: Create an Agent

```
User: "Create a math tutor"
System: "Agent 'MathTutor' created successfully!"
```

### Step 2: Select Agent

Select "MathTutor" from dropdown

### Step 3: Have Multiple Conversations

```
Conversation 1: "What's calculus?"
Conversation 2: "Explain derivatives"
Conversation 3: "How do I integrate?"
...
Conversation 10: [Agent generates insight!]
```

### Step 4: Observe Learning

**After 5 conversations:**
```
📚 5 convos | math
```

**After 10 conversations:**
```
📚 10 convos | math, calculus
💡 I've been reflecting on our conversations...
```

**After 25 conversations:**
```
📚 25 convos | math, calculus, algebra
[Agent knows your preferences and adapts responses]
```

---

## 🔍 Under the Hood

### Database Schema

**conversations table:**
| Column         | Type    | Description                |
|----------------|---------|----------------------------|
| id             | INTEGER | Primary key                |
| agent_id       | TEXT    | Agent identifier           |
| user_message   | TEXT    | User's prompt              |
| agent_response | TEXT    | Agent's reply              |
| timestamp      | TEXT    | ISO timestamp              |
| session_id     | TEXT    | Session identifier         |
| tokens_used    | INTEGER | Context tokens consumed    |

**agent_insights table:**
| Column    | Type    | Description                     |
|-----------|---------|--------------------------------|
| id        | INTEGER | Primary key                    |
| agent_id  | TEXT    | Agent identifier               |
| insight   | TEXT    | Proactive message              |
| timestamp | TEXT    | When generated                 |
| delivered | BOOLEAN | Whether shown to user          |

### File Structure

```
Agent2Agent-main/SV2B/
├── agent_conversations.db      # SQLite database
├── agent_wisdom/               # Wisdom directory
│   ├── MathMaster_wisdom.json
│   ├── CodeHelper_wisdom.json
│   └── DataBot_wisdom.json
├── agents.json                 # Agent definitions
└── main.py                     # Backend code
```

### Learning Algorithm

**Topic Extraction:**
```python
def _extract_topics(conversations):
    keywords = ["python", "javascript", "data", "math", "code", ...]
    for user_msg, agent_resp in conversations:
        combined = user_msg + " " + agent_resp
        for keyword in keywords:
            if keyword in combined.lower():
                topics.append(keyword)
    return unique(topics)[:5]  # Keep top 5
```

**Insight Generation:**
```python
def _generate_insight(agent_id, conversations):
    if len(conversations) >= 10 and len(conversations) % 10 == 0:
        return f"I've been reflecting on our {len(conversations)} conversations..."
    return None
```

---

## ⚙️ Configuration

### Adjust Learning Interval

Edit `main.py` line 328:

```python
self.learning_interval = 300  # Seconds (default: 5 minutes)

# Change to 60 for every minute:
self.learning_interval = 60

# Change to 600 for every 10 minutes:
self.learning_interval = 600
```

### Adjust Insight Polling

Edit `ChatWindow.jsx` line 71:

```javascript
const interval = setInterval(checkInsights, 30000); // ms

// Poll every minute:
const interval = setInterval(checkInsights, 60000);

// Poll every 10 seconds:
const interval = setInterval(checkInsights, 10000);
```

### Adjust Context Limit Warning

Edit `main.py` line 586:

```python
if tokens_used > 1800:  # 90% of 2048

# Warn at 80%:
if tokens_used > 1638:

# Warn at 95%:
if tokens_used > 1945:
```

### Add Custom Keywords

Edit `main.py` line 387:

```python
keywords = ["python", "javascript", "data", "math", "code", ...]

# Add your keywords:
keywords = ["python", "AI", "machine learning", "blockchain", ...]
```

---

## 💡 Advanced Features

### Conversation Patterns

Agents can detect:
- **Question frequency**: How often user asks questions
- **Topic focus**: Which topics dominate conversations
- **Response preferences**: Length, style, depth preferred
- **Time patterns**: When user typically chats

### Future Enhancements

Coming soon:
- ⏳ **Automatic summarization** at context limits
- ⏳ **Sentiment analysis** of conversations
- ⏳ **Personalized recommendations** based on patterns
- ⏳ **Multi-agent collaboration** (agents learning from each other)
- ⏳ **Export conversation history** as PDF/JSON
- ⏳ **Conversation analytics dashboard**
- ⏳ **Custom learning rules** per agent

---

## 🐛 Troubleshooting

### "No insights appearing"

**Causes:**
- Agent needs 10+ conversations
- Background learner not running
- Polling interval too long

**Solutions:**
1. Check backend logs: "Background learning system started"
2. Chat with agent 10+ times
3. Wait 5-10 minutes for learning cycle
4. Refresh frontend

### "Context warning always showing"

**Causes:**
- Very long conversation history
- Large system instructions
- Accumulated wisdom is large

**Solutions:**
1. Start new chat session
2. Shorter messages
3. Clear conversation history

### "Agent not learning topics"

**Causes:**
- Keywords not in predefined list
- Conversations too short
- Database not writable

**Solutions:**
1. Add keywords to `main.py` line 387
2. Have longer, topic-focused conversations
3. Check file permissions on database

### "Database locked" errors

**Causes:**
- Multiple processes accessing database
- File system permissions

**Solutions:**
1. Restart backend
2. Check database file permissions
3. Only run one backend instance

---

## 📈 Success Metrics

**How to know learning is working:**

✅ **Conversation counter increases** in header badge
✅ **Expertise areas appear** after 3-5 conversations
✅ **Insights generated** every 10 conversations
✅ **Context tokens displayed** and updating
✅ **Backend logs** show "Background learning system started"
✅ **wisdom.json files** created in agent_wisdom/
✅ **Database file** created: agent_conversations.db

---

## 🎉 Real-World Example

### Day 1: First Conversation

```
User: "Create a Python helper"
System: "Agent 'PythonExpert' created!"

[Select PythonExpert]
Header: PythonExpert
        📚 0 convos

User: "How do I use list comprehensions?"
Agent: "List comprehensions are..."
Header: 📚 1 convo

User: "What about dictionaries?"
Agent: "Dictionaries are key-value pairs..."
Header: 📚 2 convos
```

### Day 1: After 10 Conversations

```
Header: 📚 10 convos | python, code

[30 seconds later...]
💡 I've been reflecting on our 10 conversations,
and I think I can help you even better now!
```

### Day 2: Return to Agent

```
[Select PythonExpert]
Header: 📚 15 convos | python, code, programming

User: "Tell me about classes"
Agent: [Responds with context from previous discussions]
        "Building on what we discussed about functions..."
```

### Day 7: Experienced Agent

```
Header: 📚 45 convos | python, code, programming, OOP, data

Agent has learned:
- Your preferred explanation style
- Topics you've mastered
- Areas where you need more detail
- When you typically ask questions
```

**The agent is now truly personalized to YOU!**

---

## 🔐 Privacy & Data

### What's Stored

- ✅ Conversation text (user messages + agent responses)
- ✅ Timestamps
- ✅ Token usage
- ❌ No user identification
- ❌ No IP addresses
- ❌ No personal data

### Data Location

- **Local only**: Everything stored on your machine
- **SQLite database**: `agent_conversations.db`
- **JSON files**: `agent_wisdom/*.json`
- **No cloud sync**: Data never leaves your system

### Data Deletion

To reset an agent's learning:

```bash
# Delete database
rm agent_conversations.db

# Delete wisdom files
rm -rf agent_wisdom/

# Restart backend
python -m uvicorn main:app --reload
```

---

## 📝 Summary

### What This Feature Adds

**Backend:**
- 260+ lines of learning infrastructure
- SQLite database integration
- Background thread for continuous learning
- 4 new API endpoints
- Token estimation and monitoring

**Frontend:**
- Wisdom display in header
- Token counter with warnings
- Insight polling (30s interval)
- Special message styling for insights
- Real-time learning statistics

**Capabilities:**
- ✅ Stores all conversations permanently
- ✅ Learns topics and patterns
- ✅ Generates proactive insights
- ✅ Monitors context limits
- ✅ Accumulates wisdom over time
- ✅ Continuous background learning

---

**The agents are now truly intelligent and evolving!** 🧠✨

Every conversation makes them wiser. They learn your preferences, remember past discussions, and proactively share insights. The more you chat, the better they become.

---

**Last Updated:** January 12, 2026
**Feature Version:** 4.0 - Self-Learning Agent System
