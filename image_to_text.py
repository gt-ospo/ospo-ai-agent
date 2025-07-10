import sys

import ollama


if len(sys.argv) != 2:
    print(f'Usage: {sys.argv[0]} <image_path>', file=sys.stderr)
    exit(1)

MODEL = 'gemma3'
IMAGE_PATH = sys.argv[1]

messages = [
    dict(role='user', content='Describe this image.', images=[IMAGE_PATH]), # usually is a few paragraphs long
]
messages.append(ollama.chat(model=MODEL, messages=messages).message)
messages.append(dict(role='user', content='Can you summarize the image in one paragraph? No extra introduction or conclusion shall be written.'))
messages.append(ollama.chat(model=MODEL, messages=messages).message)
for message in messages:
    print(message, file=sys.stderr)
print(messages[-1].content)
