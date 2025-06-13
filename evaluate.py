import os
import sys
import csv

import ollama

from rag import load_text, ask


generate_model = os.environ.get("MODEL", "llama3.2:3b")
embed_model = "nomic-embed-text"
print("loading documents...", file=sys.stderr)
collection = load_text(
    embed_model=embed_model,
    embeddings_path="embeddings",
    text_path="summaries",
)

def evaluate(question: str, source_docs: str, question_type: str, source_chunk_type: str, answer: str):
    epilogue = "Focus on financial figures such as net sales, operating expenses, and losses."
    my_answer = ask(question + epilogue, collection, embed_model=embed_model, generate_model=generate_model)
    print('='*30)
    print(my_answer)
    print('-'*30)
    print(answer)
    output = ollama.generate(
        generate_model,
        prompt="Respond whether the given answer is correct. Answer with a single YES or NO at the end of your response.\n" \
        + f"Correct Answer:\n{answer}" \
        + f"Attempt:\n{my_answer}",
    )["response"]
    print(output)
    print('='*30)
    correct = output.lower().find('yes') > output.lower().find('no')
    return correct

if __name__ == '__main__':
    rows = csv.reader(sys.stdin)
    n_correct = 0
    n_total = 0
    header = next(rows, None)
    for row in rows:
        print(row[0])
        correct = evaluate(*row)
        n_correct += int(correct)
        n_total += 1
    print(f"{n_correct}/{n_total} correct")