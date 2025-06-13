import ollama

for m in ollama.list()["models"]:
    print(f"- {m.model}")

models = 'nomic-embed-text qwen3:4b qwen3:8b llama3.2:3b qwen3:14b'.split()
for model in models:
    print(model)
    ollama.pull(model)
