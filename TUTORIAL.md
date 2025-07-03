# Tutorial

This tutorial will walk you through making a simple RAG-based chatbot/agent using PACE ICE's compute resources.

Although this tutorial is applicable to many types of source datasets (i.e. data used as ground truths by the AI agent), we will follow along with an example that makes a chatbot/agent for public SEC filings.

## Prerequisites

- Python: basic knowledge (for loops, method calls, generators, etc)
- Bash: basic knowledge (for loops, variable substitution, etc)
- access to [PACE ICE's instance of Open OnDemand](https://ondemand-ice.pace.gatech.edu)
  - If off-campus, use the GT VPN (or in-browser VPN) to connect

## Step 1. Get Started with ICE

We are going to be running LLMs (large language models) on ICE.
The easiest way to get started with LLMs on ICE is to go to Open OnDemand, click "Interactive Apps" on the top navbar, and then "Ollama + Jupyter (Beta)."
![image of a dropdown of Interactive Apps, showing Ollama as the 4th option from the top](https://github.com/user-attachments/assets/8d4b9deb-90f4-48f6-98b6-c1e6b163cbd0)

Select "Temporary directory" as your Ollama models directory.
This directory is where the LLMs will be downloaded to. Since "PACE shared models" does not allow downloading additional models, we will use "Temporary directory" and download the models ourselves.
![dropdown for choosing Ollama models directory](https://github.com/user-attachments/assets/1397c880-1892-4987-b718-eb6d1c1eeb48)

For the node type, select "NVIDIA GPU (first avail)." This will ensure you get a GPU of some sort, while making sure you won't wait too long for a specific GPU to free up.

The default values can be used for everything else.

Once you click submit, you should see a card like the following:
![card showing queued job status](https://github.com/user-attachments/assets/9503a326-2f65-4dea-b04f-317a125c4d01)
This card will become green once the environment is ready:
![card showing running job status, with a connect to Jupyter button at the bottom](https://github.com/user-attachments/assets/79c02363-b076-402c-851f-aa118504e6ce)
Click "Connect to Jupyter" to open a new tab with your new environment.

## Step 2. Setting up your Environment

1. Clone this repository, and `cd` into it
2. Run `python -m venv venv` to setup a virtual enviroment, which will make installation Python packages easier.
3. Run `source venv/bin/activate` to activate the virtual environment. Now, commands like pip only affect this virtual environment, and should have no effect on your other projects (if any).
4. Run `pip install ollama nltk numpy chromadb`
  - `ollama` - used to run LLM models on ICE
  - `nltk` - used to divide the source text into sentences
  - `numpy` - convenient math subroutines
  - `chromadb` - simple vector database suited for prototyping (cf. [vector database comparison](https://github.com/gt-ospo/vsip-summer-2025-projects/blob/516848e0aef4465d0e666b573fa39767aa79d755/project-updates/ai-agent/Ken_Shibata.md#week-of-2025-jun-13))
5. (This will take a few ten minutes.) Run `python pull_models.py` to download a few models, so you can experiment and see which suits your use case the best.

## Step 3. Consider your Source Data

Listed below are a few questions to answer about your source dataset.

- What format is it in?
  - Examples: video, audio, PDF, plain text
- What format is the information in?
  - Is there a lot of template text (e.g. headers, legal notices)?
  - Is all data the same format (e.g. all plain text), or is there multiple (e.g. a graph and some text)

Based on anecdotal evidence, an LLM is more likely to hallucinate if given a *lot* of unnecessary information. Therefore, to make a good RAG chatbot, we want to give the LLM the minimum amount of information that still results in a good answer.

## Step 4. Consider how to Partition and Retrieve your Source Data

- Retrieval
  - Vector DB
  - plain text
