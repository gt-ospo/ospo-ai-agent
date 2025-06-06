import ollama


response = ollama.chat(
    model='llama3.2:latest', 
    messages=[
        {
            'role': 'user',
            'content': 'Hello! How are you today?'
        }
    ]
)

print(response)
