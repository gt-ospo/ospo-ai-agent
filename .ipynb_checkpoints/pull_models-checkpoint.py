import ollama

for m in ollama.list()["models"]:
    print(f"- {m.model}")

models = 'nomic-embed-text llama3.1:8b llama3.2:1b llama3.2:3b llama3.3 mistral qwen3:0.6b qwen3:1.7b qwen3:4b qwen3:8b'.split()
for model in models:
    print(model)
    ollama.pull(model)
