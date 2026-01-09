# Model Management Guide

## 🎯 Overview

The Agent2Agent platform now includes **comprehensive model management** directly from the UI! No more manual downloads, command-line tools, or environment variables needed.

### Key Features
- ✅ Download models from HuggingFace with one click
- ✅ Real-time download progress bars
- ✅ Switch between models instantly
- ✅ Add custom model URLs from anywhere
- ✅ Delete unused models to save space
- ✅ Visual status indicators (downloaded/active)
- ✅ Background downloads (non-blocking)

---

## 🚀 Quick Start

### **Step 1: Open Model Manager**

1. Start your backend and frontend:
   ```cmd
   # Terminal 1 - Backend
   cd Agent2Agent-main\SV2B
   python -m uvicorn main:app --reload

   # Terminal 2 - Frontend
   cd Agent2Agent-main\SV2\agent-ui
   npm run dev
   ```

2. In the UI, click **🤖 Models** button in the sidebar (bottom section)

### **Step 2: Download a Model**

1. You'll see 3 default models:
   - **Mistral 7B Instruct Q4_K_M** (4.4 GB) - Balanced quality/speed
   - **Mistral 7B Instruct Q2_K** (3.0 GB) - Smaller, faster
   - **Llama 2 7B Chat Q4_K_M** (4.1 GB) - Meta's chat model

2. Click **Download** button on your preferred model
3. Watch the progress bar fill up (downloads in background)
4. First download takes 5-15 minutes depending on internet speed

### **Step 3: Activate the Model**

1. Once download completes, click **Select** button
2. Model loads into memory (takes 10-30 seconds)
3. Green checkmark ✓ appears when active
4. Start chatting with agents using this model!

---

## 📋 Available Models (Default)

### 1. **Mistral 7B Instruct Q4_K_M** (Recommended)
- **Size**: 4.4 GB download, ~5 GB RAM when loaded
- **Quality**: High
- **Speed**: Fast
- **Best for**: General use, agent creation, conversations
- **URL**: `https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf`

### 2. **Mistral 7B Instruct Q2_K** (Lightweight)
- **Size**: 3.0 GB download, ~3.5 GB RAM when loaded
- **Quality**: Medium
- **Speed**: Very Fast
- **Best for**: Low-RAM systems, quick testing, simple tasks
- **URL**: `https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q2_K.gguf`

### 3. **Llama 2 7B Chat Q4_K_M** (Meta Official)
- **Size**: 4.1 GB download, ~4.5 GB RAM when loaded
- **Quality**: High
- **Speed**: Fast
- **Best for**: Conversational AI, safety-focused applications
- **URL**: `https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf`

---

## 🎨 UI Walkthrough

### Model Card Indicators

**Not Downloaded:**
```
┌─────────────────────────────┐
│ Mistral 7B Instruct Q4_K_M  │
│ Balanced performance...      │
│ Size: 4.4 GB                 │
│ [Download]                   │
└─────────────────────────────┘
```

**Downloading:**
```
┌─────────────────────────────┐
│ Mistral 7B Instruct Q4_K_M  │
│ Balanced performance...      │
│ Size: 4.4 GB                 │
│ Downloading: 67.3%           │
│ ████████████░░░░░░░░         │
└─────────────────────────────┘
```

**Downloaded (Not Active):**
```
┌─────────────────────────────┐
│ Mistral 7B Instruct Q4_K_M  │
│ Balanced performance...      │
│ Size: 4.4 GB | ✓ Downloaded │
│ [Select] [Delete]            │
└─────────────────────────────┘
```

**Active Model (Green Border):**
```
╔═════════════════════════════╗ (green)
║ Mistral 7B Instruct Q4_K_M ✓║
║ Balanced performance...      ║
║ Size: 4.4 GB | ✓ Downloaded ║
║ ● Active                     ║
╚═════════════════════════════╝
```

---

## 🔧 Adding Custom Models

### From HuggingFace

1. Click **+ Add Custom Model URL**
2. Enter the direct GGUF URL, for example:
   ```
   https://huggingface.co/TheBloke/CodeLlama-13B-Instruct-GGUF/resolve/main/codellama-13b-instruct.Q4_K_M.gguf
   ```
3. Click **Add Model**
4. Model appears in your list
5. Click **Download** to fetch it

### Finding Models

**Popular GGUF Sources:**
- TheBloke on HuggingFace: https://huggingface.co/TheBloke
- Search GGUF models: https://huggingface.co/models?library=gguf

**Direct URL Format:**
```
https://huggingface.co/{author}/{model-repo}/resolve/main/{filename}.gguf
```

**Example URLs:**
```bash
# CodeLlama 13B
https://huggingface.co/TheBloke/CodeLlama-13B-Instruct-GGUF/resolve/main/codellama-13b-instruct.Q4_K_M.gguf

# Mixtral 8x7B
https://huggingface.co/TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF/resolve/main/mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf

# Phi-2
https://huggingface.co/TheBloke/phi-2-GGUF/resolve/main/phi-2.Q4_K_M.gguf
```

---

## 💾 Storage Management

### Where Models Are Stored

**Backend Directory:**
```
Agent2Agent-main/SV2B/models/
├── mistral-7b-instruct-v0.2.Q4_K_M.gguf (4.4 GB)
├── mistral-7b-instruct-v0.2.Q2_K.gguf   (3.0 GB)
└── llama-2-7b-chat.Q4_K_M.gguf          (4.1 GB)
```

### Deleting Models

1. Open Model Manager
2. Find the model you want to remove
3. Click **Delete** button (only shows for inactive models)
4. Confirm deletion
5. Disk space freed immediately

**Note:** You cannot delete the currently active model. Switch to another model first, then delete.

---

## 🔄 Switching Models

### During Active Conversation

You can switch models **mid-conversation**:

1. Open Model Manager
2. Select a different downloaded model
3. Click **Select**
4. Model loads (~10-30 seconds)
5. Continue chatting with new model

**What Happens:**
- Previous model unloaded from RAM
- New model loaded into RAM
- Conversation history preserved
- Agents continue working with new model

### Performance Tip

Keep only 1-2 models downloaded if disk space is limited. You can always re-download later!

---

## 📊 System Requirements

### RAM Requirements

| Model Quantization | Downloaded Size | RAM When Loaded |
|--------------------|-----------------|-----------------|
| Q2_K               | ~3 GB           | ~3.5 GB         |
| Q3_K               | ~3.5 GB         | ~4 GB           |
| Q4_K_M             | ~4-4.5 GB       | ~5 GB           |
| Q5_K_M             | ~5-5.5 GB       | ~6 GB           |
| Q8_0               | ~7-7.5 GB       | ~8.5 GB         |

### Disk Space

- **Minimum**: 5 GB free (1 model)
- **Recommended**: 15+ GB free (3-4 models)

### Internet Speed

| Speed      | 4 GB Model Download Time |
|------------|--------------------------|
| 10 Mbps    | ~50 minutes              |
| 50 Mbps    | ~10 minutes              |
| 100 Mbps   | ~5 minutes               |
| 500 Mbps   | ~1 minute                |

---

## 🔍 Troubleshooting

### "Download Failed"

**Causes:**
- Network interruption
- Insufficient disk space
- Invalid URL

**Solutions:**
1. Check internet connection
2. Verify 5+ GB free disk space
3. Try downloading again
4. Test URL in browser first

### "Failed to Load Model"

**Causes:**
- Insufficient RAM
- Corrupted download

**Solutions:**
1. Close other applications to free RAM
2. Delete and re-download the model
3. Try a smaller quantization (Q2_K instead of Q4_K_M)

### "Model Not Responding"

**Causes:**
- Model still loading
- System overloaded

**Solutions:**
1. Wait 30-60 seconds for large models
2. Check backend terminal for "Model loaded successfully!"
3. Restart backend if necessary

### Download Stuck at 0%

**Causes:**
- Firewall blocking
- HuggingFace rate limiting

**Solutions:**
1. Check firewall settings
2. Wait 5 minutes and try again
3. Use VPN if location-blocked

---

## 🎓 Advanced Usage

### Custom Model Configuration

Edit `main.py` to customize model loading parameters:

```python
# Line 92-97
llm = Llama(
    model_path=str(model_path),
    n_ctx=2048,        # Increase for longer conversations
    n_threads=4,       # Increase for faster inference
    n_gpu_layers=0,    # Set >0 for GPU acceleration
)
```

### GPU Acceleration

If you have an NVIDIA GPU:

1. **Install CUDA version of llama-cpp-python:**
   ```cmd
   CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
   ```

2. **Update main.py line 96:**
   ```python
   n_gpu_layers=35,  # Offload 35 layers to GPU
   ```

3. **Restart backend** to see 5-10x speed improvement!

### API Usage (Programmatic)

**List Models:**
```bash
curl http://localhost:8000/models
```

**Download Model:**
```bash
curl -X POST "http://localhost:8000/models/download?model_id=mistral-7b-instruct-q4"
```

**Check Progress:**
```bash
curl http://localhost:8000/models/download-progress/mistral-7b-instruct-q4
```

**Select Model:**
```bash
curl -X POST "http://localhost:8000/models/select?model_id=mistral-7b-instruct-q4"
```

---

## 📈 Best Practices

### 1. **Start with Q4_K_M**
Balanced quality and speed for most users.

### 2. **Download During Off-Hours**
Large downloads work best overnight or during low network usage.

### 3. **Test Before Committing**
Download Q2_K first to test quickly, then upgrade to Q4_K_M if satisfied.

### 4. **One Active, Two Backup**
Keep 1 model loaded, 2 downloaded as alternatives.

### 5. **Regular Cleanup**
Delete unused models monthly to free disk space.

---

## 🆕 What's New

This feature adds:
- ✅ **6 new API endpoints** for model management
- ✅ **ModelManager.jsx** - 400+ line UI component
- ✅ **Real-time progress tracking** with WebSocket-like polling
- ✅ **Background downloads** using FastAPI BackgroundTasks
- ✅ **Multi-model support** - load/unload dynamically
- ✅ **Custom model URLs** - not limited to defaults
- ✅ **Thread-safe operations** with locking mechanisms

---

## 🔮 Future Enhancements

Planned features:
- ⏳ Model quantization comparison tool
- ⏳ Automatic model recommendations based on RAM
- ⏳ Model performance benchmarks
- ⏳ One-click model updates
- ⏳ Cloud storage integration
- ⏳ Model fine-tuning interface

---

## 📞 Support

**Common Questions:**

**Q: Can I use models from other sources?**
A: Yes! Any GGUF file accessible via HTTP/HTTPS works.

**Q: Do I need Ollama anymore?**
A: No! This system completely replaces Ollama.

**Q: Can I run multiple models simultaneously?**
A: Only one model can be active at a time (RAM limitation).

**Q: How do I update a model?**
A: Delete the old version and download the new one.

**Q: Are models shared between users?**
A: Yes, downloaded models are stored globally and shared.

---

## 🎉 Summary

**Before this feature:**
- ❌ Manual model downloads
- ❌ Command-line configuration
- ❌ Environment variable setup
- ❌ Ollama dependency

**After this feature:**
- ✅ One-click downloads
- ✅ Visual progress tracking
- ✅ Easy model switching
- ✅ No external tools needed

**The Agent2Agent platform is now fully self-contained with professional model management!**

---

**Last Updated:** January 2026
**Feature Version:** 3.0 - Model Management Release
