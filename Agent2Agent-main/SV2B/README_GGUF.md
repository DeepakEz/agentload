# GGUF Model Loading Guide

This backend now supports loading GGUF models directly from any URL, eliminating the need for Ollama.

## Features

- **Download from anywhere**: Load GGUF models from any HTTP/HTTPS URL
- **Automatic caching**: Models are cached locally after first download
- **No Ollama required**: Direct model inference using llama-cpp-python
- **Configurable**: Easy to switch models via environment variables

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Model (Optional)

You can configure the model URL and cache directory using environment variables:

```bash
# Set custom model URL
export MODEL_URL="https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf"

# Set custom cache directory
export MODEL_CACHE_DIR="./my_models"
```

**Default model**: Mistral-7B-Instruct-v0.2 (Q4_K_M quantization)

### 3. Run the Server

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## Using Different Models

### From HuggingFace

Most GGUF models on HuggingFace can be loaded. Find models at:
- https://huggingface.co/TheBloke (popular quantized models)
- https://huggingface.co/models?library=gguf

Example URLs:
```bash
# Llama 2 7B Chat
export MODEL_URL="https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf"

# Mixtral 8x7B Instruct
export MODEL_URL="https://huggingface.co/TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF/resolve/main/mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf"

# CodeLlama 13B
export MODEL_URL="https://huggingface.co/TheBloke/CodeLlama-13B-Instruct-GGUF/resolve/main/codellama-13b-instruct.Q4_K_M.gguf"
```

### From Local File

You can also use a local GGUF file:

```bash
# Download model manually
wget https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf -P ./models/

# The server will detect and use it automatically
```

### From Any URL

Any publicly accessible GGUF file URL will work:

```bash
export MODEL_URL="https://example.com/path/to/model.gguf"
```

## Model Quantization Guide

GGUF models come in different quantization levels (Q2, Q3, Q4, Q5, Q6, Q8):

- **Q2_K**: Smallest size, lowest quality (~2-3 GB for 7B models)
- **Q4_K_M**: Good balance of size and quality (~4 GB for 7B models) **[Recommended]**
- **Q5_K_M**: Better quality, larger size (~5 GB for 7B models)
- **Q8_0**: Highest quality, largest size (~7 GB for 7B models)

Choose based on your available RAM and quality requirements.

## GPU Acceleration (Optional)

To enable GPU acceleration, edit `main.py` line 76:

```python
llm_model = Llama(
    model_path=str(model_path),
    n_ctx=2048,
    n_threads=4,
    n_gpu_layers=35,  # Change from 0 to number of layers (e.g., 35 for 7B models)
)
```

Requires CUDA-enabled `llama-cpp-python`:
```bash
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
```

## Configuration Options

Edit `main.py` to customize model parameters:

```python
# Line 74-77: Model loading configuration
llm_model = Llama(
    model_path=str(model_path),
    n_ctx=2048,        # Context window size (increase for longer conversations)
    n_threads=4,       # CPU threads (increase for faster inference)
    n_gpu_layers=0,    # GPU layers (set >0 for GPU acceleration)
)

# Line 85-91: Inference parameters
response = llm(
    prompt,
    max_tokens=512,     # Maximum response length
    temperature=0.7,    # Creativity (0.0-1.0, higher = more creative)
    top_p=0.9,          # Nucleus sampling (0.0-1.0)
    echo=False,         # Don't echo the prompt in response
    stop=["</s>", "User:", "\n\n\n"]  # Stop sequences
)
```

## Troubleshooting

### Out of Memory
- Use smaller quantization (Q2_K or Q3_K)
- Reduce `n_ctx` value
- Close other applications

### Slow inference
- Increase `n_threads` to match CPU cores
- Enable GPU acceleration
- Use smaller model

### Download fails
- Check internet connection
- Verify URL is accessible
- Try downloading manually and place in cache directory

## API Usage

The API remains unchanged from the Ollama version:

```bash
curl -X POST http://localhost:8000/process_message \
  -H "Content-Type: application/json" \
  -d '{"user_prompt": "Hello, how are you?"}'
```

Response:
```json
{
  "response": "I'm doing well, thank you for asking! How can I help you today?"
}
```
