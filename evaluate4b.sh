set -euxo pipefail

run() {
    export MODEL="$1"
    python evaluate.py < docs/qna_data.csv > "results4-chunked-rag/$MODEL"
}

run llama3.1:8b
run llama3.2:1b
run llama3.2:3b
run llama3.3
# run mistral
# run qwen3:0.6b
# run qwen3:1.7b
# run qwen3:4b
# run qwen3:8b
