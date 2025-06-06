summaries/%.txt: text/%.txt
	python summarize.py "$<" "$@"

summaries_all:
	bash -c 'for file in docs/*.pdf; do file2="$$(basename "$$file")"; make "summaries/$${file2%.*}.txt"; done'

.PHONY: summaries_all

embeddings/%.json: summaries/%.txt
	python embed.py "$<" "$@"

embeddings_all:
	bash -c 'for file in docs/*.pdf; do file2="$$(basename "$$file")"; make "embeddings/$${file2%.*}.json"; done'

.PHONY: embeddings_all

text: docs
	mkdir -p text
	bash -c 'for file in docs/*.pdf; do file2="$$(basename "$$file")"; pdftotext "$$file" "text/$${file2%.*}.txt"; done'

docs:
	./get-pdfs.sh

clean:
	rm -rf docs
	rm -rf text
	rm -rf embeddings

.PHONY: clean
