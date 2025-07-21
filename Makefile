TUTORIAL_WITH_CODE.ipynb: TUTORIAL_WITH_CODE.md
	jupytext --to ipynb $< --output $@
