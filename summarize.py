import itertools
import json
import sys

import ollama


text_path = sys.argv[1]
summary_path = sys.argv[2]
with open(text_path) as f:
    text = f.read()
chunks = list(itertools.batched(text, 8000))
full_summary = ""
for i, chunk in enumerate(chunks):
    chunk = ''.join(chunk)
    print(f"summarizing {i+1}/{len(chunks)}...")
    summary = ollama.generate(
        model="llama3.2",
        prompt=f"Make a short self-contained summary of the following text from {text_path} in Markdown: {chunk}\n\n If there is not much to say, just leave it blank. Make sure the summary is in Markdown and is easy to understand.",
    )["response"]
    print(f"summarized {text_path}.", file=sys.stderr)
    print("vvv summary")
    print(summary)
    print("^^^ summary")
    relevant = ollama.generate(
        model="llama3.2",
        prompt=f"Does the following contain useful information (i.e. not just formatting)?\n{summary} " + "Answer in JSON format, with a single key \"relevant\" such as {\"relevant\": true} or {\"relevant\": false}. If no summary is provided, return {\"relevant\": false}",
    )["response"]
    print(relevant)
    if "true" not in relevant.lower() or "false" in relevant.lower():
        print("skipped", file=sys.stderr)
        continue
    full_summary += summary
print("vvv full summary")
print(full_summary)
print("^^^ full summary")
with open(summary_path, "w") as f:
    f.write(full_summary)
print(f"embedded {text_path} to {summary_path}.", file=sys.stderr)

