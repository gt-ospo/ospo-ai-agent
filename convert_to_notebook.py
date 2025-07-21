#!/usr/bin/env python3
"""
Convert TUTORIAL_WITH_CODE.md to TUTORIAL_WITH_CODE.ipynb using pypandoc.

The Markdown format is easier to edit and track changes with Git,
while the notebook format is easier to run (e.g. on Colab or PACE ICE).
"""

import os
import sys
import pypandoc


def main():
    """Main function"""
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input_file> <output_file>")
        exit(1)
    pypandoc.convert_file(sys.argv[1], "ipynb", outputfile=sys.argv[2])

if __name__ == "__main__":
    main()
