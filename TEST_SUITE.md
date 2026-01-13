# Agent2Agent - Comprehensive Test Suite

Complete test prompts to verify all claimed functionalities of the self-learning AI agent system.

---

## 🎯 Test Environment Setup

**Before testing, ensure:**
1. ✅ Backend running: `python -m uvicorn main:app --reload`
2. ✅ Frontend running: `npm run dev`
3. ✅ Model downloaded and selected (Mistral 7B Q4 recommended)
4. ✅ Browser console open (F12) to see logs

---

## Test 1: Direct GGUF Model Loading

**Claimed:** Load GGUF models directly without Ollama

### Test Steps:
1. Click **🤖 Models** button
2. Verify you see 3 default models listed
3. Click **Download** on "Mistral 7B Instruct Q4_K_M"
4. Watch progress bar (should show percentage)
5. When complete, click **Select**

**Expected Results:**
- ✅ Model downloads to `SV2B/models/` directory
- ✅ Progress bar shows real-time download progress
- ✅ "Select" button becomes clickable after download
- ✅ Model loads without Ollama process

**Pass/Fail:** _____

**Backend Terminal Should Show:**
```
Model loaded: mistral-7b-instruct-q4
```

---

## Test 2: AI-Generated Agent Creation

**Claimed:** Describe agent in plain English, AI generates name and instructions

### Test Prompts:

**Test 2A - Python Coding Assistant:**
```
I want an agent that helps me write clean Python code, debug errors, and suggests best practices
```

**Expected Results:**
- ✅ AI generates agent name like "PythonCodeHelper" or "PythonMaster"
- ✅ System instruction includes Python expertise, debugging, best practices
- ✅ Can edit both name and instruction before creating
- ✅ Agent appears in dropdown after creation

**Pass/Fail:** _____

**Test 2B - Creative Writing Partner:**
```
Create an agent for creative writing, story ideas, and character development
```

**Expected Results:**
- ✅ Agent name like "CreativeWriter" or "StoryWeaver"
- ✅ Instructions mention creativity, stories, characters
- ✅ Different from previous agent's instructions

**Pass/Fail:** _____

**Test 2C - Math Tutor:**
```
I need help with algebra, calculus, and explaining math concepts step-by-step
```

**Expected Results:**
- ✅ Agent name like "MathTutor" or "AlgebraMaster"
- ✅ Instructions include step-by-step explanations

**Pass/Fail:** _____

---

## Test 3: Conversation Persistence

**Claimed:** Conversations persist across page refreshes

### Test Steps:

**Test 3A - Basic Persistence:**
1. Select any agent
2. Send 3 messages:
   - "Hello, my name is Alex"
   - "I'm learning Python programming"
   - "My goal is to build a web scraper"
3. Refresh page (F5)
4. Check if all 3 messages appear

**Expected Results:**
- ✅ All 3 user messages visible
- ✅ All 3 agent responses visible
- ✅ Same agent is selected automatically
- ✅ Messages in correct order

**Pass/Fail:** _____

**Test 3B - Multi-Agent Persistence:**
1. Chat with Agent A (3 messages)
2. Switch to Agent B (3 messages)
3. Refresh page
4. Agent B should be selected with its 3 messages
5. Switch to Agent A
6. Agent A's 3 messages should load

**Expected Results:**
- ✅ Each agent has separate conversation history
- ✅ Switching agents loads correct history
- ✅ localStorage remembers last selected agent

**Pass/Fail:** _____

---

## Test 4: Self-Learning System

**Claimed:** Agents learn from conversations and accumulate wisdom

### Test Conversation Sequence:

**Send these 10 messages to test learning:**

1. "I'm interested in machine learning"
2. "What are neural networks?"
3. "Explain backpropagation"
4. "How does gradient descent work?"
5. "What's the difference between supervised and unsupervised learning?"
6. "Tell me about convolutional neural networks"
7. "How do I prevent overfitting?"
8. "What are activation functions?"
9. "Explain dropout regularization"
10. "What's transfer learning?"

**Expected Results After 10 Conversations:**
- ✅ Header shows: "📚 10 convos"
- ✅ Expertise areas include: "machine learning", "neural networks", or similar
- ✅ Agent's wisdom stored in `SV2B/agent_wisdom/{agent_name}_wisdom.json`
- ✅ Database file exists: `SV2B/agent_conversations.db`

**Verify Database:**
```bash
sqlite3 agent_conversations.db "SELECT COUNT(*) FROM conversations WHERE agent_id='YourAgentName';"
```
Should return: 10

**Pass/Fail:** _____

---

## Test 5: AI-Powered Background Thinking

**Claimed:** Agent thinks in background and generates creative insights

### Test Steps:

1. Have 5+ conversations with an agent (use Test 4 messages)
2. Wait 5 minutes (background learning interval)
3. Check backend terminal for:
   ```
   💡 Agent {name} discovered: [insight text]
   ```
4. Within 30 seconds, check if green insight message appears in UI

**Expected Insight Examples:**
- Pattern recognition: "I noticed you're particularly interested in deep learning architectures..."
- Novel suggestion: "Based on our discussions about neural networks, you might find GAN architectures interesting..."
- Creative connection: "Your questions about overfitting and regularization suggest you're building a model..."

**Expected Results:**
- ✅ Backend logs show insight generation
- ✅ Green message with 💡 icon appears in chat
- ✅ Insight is relevant to conversation topics
- ✅ Insight is NOT generic ("I've been reflecting..." ❌)

**Pass/Fail:** _____

**Check Wisdom File:**
```bash
cat SV2B/agent_wisdom/{agent_name}_wisdom.json
```
Should show extracted expertise areas like: `["machine-learning", "neural-networks"]`

---

## Test 6: Context Window Management

**Claimed:** Monitors token usage and warns at 90% limit

### Test Steps:

1. Select an agent
2. Send a very long message (500+ words)
3. Check header for token counter
4. Continue sending long messages until counter > 1800

**Expected Results:**
- ✅ Token counter visible: "Context: X / 2048 tokens"
- ✅ When > 1800: "⚠️ Near limit" warning appears
- ✅ Counter updates after each message

**Pass/Fail:** _____

---

## Test 7: Proactive Agent Engagement

**Claimed:** Agent initiates conversations with discoveries

### Test Steps:

1. Have 10+ conversations about a specific topic (e.g., Python)
2. Wait 5+ minutes for background analysis
3. Observe for proactive messages

**Expected Results:**
- ✅ Agent sends message without user prompt
- ✅ Message has green background and 💡 icon
- ✅ Message is contextually relevant
- ✅ Appears in middle of chat timeline, not just at the end

**Pass/Fail:** _____

---

## Test 8: Error Handling & Stability

**Claimed:** Robust error handling with helpful messages

### Test 8A - No Model Loaded:
1. Don't select any model
2. Try to send a message

**Expected Results:**
- ✅ Error message: "No model loaded. Please download and select a model..."
- ✅ Doesn't hang or crash

**Pass/Fail:** _____

### Test 8B - Backend Stopped:
1. Stop backend (Ctrl+C)
2. Try to send a message

**Expected Results:**
- ✅ Error message: "Cannot connect to server..."
- ✅ Helpful guidance to restart backend

**Pass/Fail:** _____

### Test 8C - Timeout:
1. Send extremely long message (2000+ words)
2. Wait for response

**Expected Results:**
- ✅ After 2 minutes: "Request timed out. The model might be too slow..."
- ✅ Send button becomes clickable again

**Pass/Fail:** _____

---

## Test 9: Model Management

**Claimed:** Download and manage multiple models

### Test Steps:

1. Open Models UI
2. Download 2 different models
3. Switch between them using "Select" button
4. Delete one model

**Expected Results:**
- ✅ Multiple models can be downloaded
- ✅ Only one model "Selected" at a time
- ✅ Switching models unloads previous, loads new
- ✅ Delete removes file from `SV2B/models/`

**Pass/Fail:** _____

---

## Test 10: Conversation History Loading

**Claimed:** Loads last 50 conversations on page load

### Test Steps:

1. Create agent and have 15+ conversations
2. Refresh page
3. Scroll up in chat

**Expected Results:**
- ✅ All 15 conversations loaded
- ✅ Messages in correct chronological order
- ✅ Both user and agent messages present
- ✅ Loading indicator shows briefly

**Pass/Fail:** _____

---

## Test 11: Agent-Specific Memory

**Claimed:** Each agent has separate conversation history

### Test Steps:

1. Create Agent A - send: "My favorite color is blue"
2. Create Agent B - send: "My favorite color is red"
3. Switch to Agent A - send: "What's my favorite color?"
4. Switch to Agent B - send: "What's my favorite color?"

**Expected Results:**
- ✅ Agent A responds: "blue"
- ✅ Agent B responds: "red"
- ✅ Agents don't mix up information
- ✅ Each maintains separate context

**Pass/Fail:** _____

---

## Test 12: Background Learning - Topic Extraction

**Claimed:** AI extracts topics intelligently, not just keywords

### Test Conversation:

Send these varied messages:
1. "I'm building a recommendation system for movies"
2. "Should I use collaborative filtering or content-based filtering?"
3. "How do I handle the cold start problem?"
4. "What's matrix factorization in this context?"
5. "Tell me about user-item matrices"

**Wait 5 minutes, then check wisdom file:**
```bash
cat SV2B/agent_wisdom/{agent_name}_wisdom.json | grep expertise
```

**Expected Results:**
- ✅ Topics like: "recommendation-systems", "collaborative-filtering", "machine-learning"
- ✅ NOT just: "python", "data" (too generic)
- ✅ Shows intelligent understanding of conversation theme

**Pass/Fail:** _____

---

## Test 13: Agent Wisdom Display

**Claimed:** Shows conversation count and expertise areas in header

### Test Steps:

1. Have 20+ conversations with an agent about specific topics
2. Refresh page
3. Check header badge

**Expected Results:**
- ✅ Shows: "📚 20 convos | topic1, topic2, topic3"
- ✅ Topics are relevant to conversations
- ✅ Updates in real-time after each conversation

**Pass/Fail:** _____

---

## Test 14: Concurrent Insights

**Claimed:** Can receive insights while actively chatting

### Test Steps:

1. Have 10+ conversations
2. Continue chatting while background learner analyzes
3. Observe if insights appear mid-conversation

**Expected Results:**
- ✅ Insight can appear between your messages
- ✅ Doesn't interrupt typing
- ✅ Automatically marked as delivered
- ✅ Doesn't duplicate

**Pass/Fail:** _____

---

## Test 15: Conversation Database Integrity

**Claimed:** All conversations stored reliably in SQLite

### Verification Commands:

```bash
# Check database exists
ls -la SV2B/agent_conversations.db

# Count total conversations
sqlite3 SV2B/agent_conversations.db "SELECT COUNT(*) FROM conversations;"

# View recent conversations
sqlite3 SV2B/agent_conversations.db "SELECT user_message, timestamp FROM conversations ORDER BY timestamp DESC LIMIT 5;"

# Check insights table
sqlite3 SV2B/agent_conversations.db "SELECT COUNT(*) FROM agent_insights;"
```

**Expected Results:**
- ✅ Database file exists
- ✅ Count matches number of conversations
- ✅ Timestamps are correct
- ✅ No duplicate entries

**Pass/Fail:** _____

---

## 🎓 Advanced Tests

### Test 16: Long Conversation Context

Send 20+ messages in a row, then ask:
```
What did we talk about at the beginning of this conversation?
```

**Expected Results:**
- ✅ Agent references earlier messages
- ✅ Shows understanding of full conversation context

**Pass/Fail:** _____

---

### Test 17: Multiple Agents Learning Different Topics

1. Agent A: 10 conversations about cooking
2. Agent B: 10 conversations about programming
3. Check each agent's wisdom file

**Expected Results:**
- ✅ Agent A expertise: cooking-related topics
- ✅ Agent B expertise: programming-related topics
- ✅ No cross-contamination

**Pass/Fail:** _____

---

### Test 18: Insight Quality Assessment

After receiving 3+ insights, rate each:
1. Is it specific to your conversations? (Yes/No)
2. Is it creative/novel? (Yes/No)
3. Is it actionable/helpful? (Yes/No)

**Pass Criteria:**
- ✅ At least 2 out of 3 insights score 3/3
- ✅ No generic template messages
- ✅ Shows genuine AI analysis

**Pass/Fail:** _____

---

### Test 19: Vector Semantic Search

**Claimed:** ChromaDB integration for semantic conversation search

### Test Steps:

1. Have 10+ conversations about Python programming
2. Use the vector search endpoint:
   ```bash
   curl "http://localhost:8000/agents/YourAgentName/vector-search?query=optimization%20advice&limit=5"
   ```
3. Verify results contain relevant conversations

**Expected Results:**
- ✅ Returns list of similar conversations
- ✅ Results are semantically relevant (not just keyword matches)
- ✅ Finds conversations about "performance" when searching for "optimization"
- ✅ Each result shows conversation text and metadata

**Alternative Test (if ChromaDB not installed):**
- ✅ Backend logs show: "⚠️ ChromaDB not installed. Vector search disabled."
- ✅ Endpoint returns error message gracefully
- ✅ System continues working normally

**Pass/Fail:** _____

---

### Test 20: Tool Registry Execution

**Claimed:** Safe tool execution framework with built-in tools

### Test Steps:

**Test 20A - List Tools:**
```bash
curl http://localhost:8000/tools
```

**Expected Results:**
- ✅ Returns list with 3 tools: calculator, text_analyzer, timestamp
- ✅ Each tool has name and description

**Test 20B - Calculator Tool:**
```bash
curl -X POST http://localhost:8000/tools/calculator/execute \
  -H "Content-Type: application/json" \
  -d '{"args": ["2 + 2 * 3"]}'
```

**Expected Results:**
- ✅ Returns: `{"tool": "calculator", "result": "8"}`
- ✅ Handles invalid input: `calculator('rm -rf /')` returns error

**Test 20C - Text Analyzer Tool:**
```bash
curl -X POST http://localhost:8000/tools/text_analyzer/execute \
  -H "Content-Type: application/json" \
  -d '{"args": ["Hello world! This is a test."]}'
```

**Expected Results:**
- ✅ Returns word count, character count, line count
- ✅ Accurate counts

**Pass/Fail:** _____

---

### Test 21: Master Architect Optimization

**Claimed:** Meta-agent optimizes other agents' wisdom

### Test Steps:

1. Create agent and have 20+ conversations
2. Manually add duplicate expertise to wisdom file (for testing):
   ```bash
   # Edit agent_wisdom/YourAgent_wisdom.json
   # Add duplicates like: ["python", "python", "ml", "ai", "machine-learning"]
   ```
3. Run optimization:
   ```bash
   curl -X POST http://localhost:8000/agents/YourAgentName/optimize
   ```
4. Check wisdom file again

**Expected Results:**
- ✅ Optimization report shows before/after counts
- ✅ Duplicates removed (["python", "python"] → ["python"])
- ✅ Similar topics consolidated (["ml", "ai"] → ["machine-learning"])
- ✅ Insights deduplicated
- ✅ Patterns pruned to last 50

**Verify with:**
```bash
curl http://localhost:8000/agents/YourAgentName/health
```
Should show health metrics and expertise count.

**Pass/Fail:** _____

---

### Test 22: Wisdom Pruning Automation

**Claimed:** Automatic pruning every 10 conversations prevents bloat

### Test Steps:

1. Create new agent
2. Manually edit wisdom file to add 20 expertise areas
3. Have exactly 10 conversations (count carefully)
4. Check backend logs for pruning message
5. Verify wisdom file now has max 15 expertise areas

**Expected Results:**
- ✅ After 10th conversation, pruning triggers automatically
- ✅ Backend logs: "Pruning wisdom for {agent_name}"
- ✅ Expertise areas limited to 15
- ✅ Duplicates removed
- ✅ Pruning happens silently (no user-facing messages)

**Verification Commands:**
```bash
# Before: 20 expertise areas
cat SV2B/agent_wisdom/YourAgent_wisdom.json | jq '.expertise_areas | length'

# After 10 conversations: 15 or fewer
cat SV2B/agent_wisdom/YourAgent_wisdom.json | jq '.expertise_areas | length'
```

**Pass/Fail:** _____

---

### Test 23: System Status Endpoint

**Claimed:** Overall system status and capabilities

### Test Steps:

```bash
curl http://localhost:8000/system/status
```

**Expected Results:**
- ✅ Shows agents_count
- ✅ Shows tools_count: 3
- ✅ Shows vector_search_enabled: true or false
- ✅ Shows background_learner_active: true
- ✅ Lists tools_available: ["calculator", "text_analyzer", "timestamp"]

**Pass/Fail:** _____

---

## 📊 Overall Test Results Summary

| Category | Tests Passed | Total Tests | Pass Rate |
|----------|--------------|-------------|-----------|
| Core Features | ___/5 | 5 | ___% |
| Learning System | ___/6 | 6 | ___% |
| Stability | ___/3 | 3 | ___% |
| Advanced Features | ___/4 | 4 | ___% |
| New Enhancements | ___/5 | 5 | ___% |
| **TOTAL** | **___/23** | **23** | **___% **|

---

## ✅ Minimum Passing Criteria

For the system to be considered "working as claimed":
- **85%+ overall pass rate** (20/23 tests)
- **100% on Tests 1-4** (core functionality)
- **At least 1 genuine AI-generated insight** (Test 5)
- **No crashes or data loss** (Test 15)
- **At least 3/5 new enhancement tests passing** (Tests 19-23)

---

## 🐛 Common Issues & Solutions

### Issue: "I couldn't load our previous conversations"
**Solution:** This is normal for brand new agents. Database creates on first conversation.

### Issue: No insights appearing
**Solution:** Need 5+ conversations + 5 minute wait. Check backend terminal for learning logs.

### Issue: Topics not extracting
**Solution:** Ensure model is loaded and backend learner is running (check startup logs).

### Issue: Database errors
**Solution:** Check `SV2B/` directory permissions. Database auto-creates if missing.

---

## 📝 Test Execution Log

**Tester Name:** _____________
**Date:** _____________
**Model Used:** _____________
**Backend Version:** _____________

**Notes:**
_______________________________________
_______________________________________
_______________________________________

**Final Verdict:** PASS / FAIL
**Overall Score:** ____%

---

**For developers:** Run this test suite before each release to ensure all claimed functionalities work correctly.
