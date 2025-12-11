from llama_cpp import Llama

llm = Llama(model_path="mistral-7b-instruct-v0.2.Q2_K.gguf", n_ctx=1024)

try:
    out = llm("Hello world", max_tokens=10)
    print(out["choices"][0]["text"])
except Exception as e:
    print("LLM failed:", e)
