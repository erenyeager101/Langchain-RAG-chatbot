# Langchain RAG Chatbot Tutorial

This tutorial explains how I built a chatbot using Langchain and Retrieval-Augmented Generation (RAG). Follow the steps below to set up the project and understand how it works.

## Step 1: Install Dependencies

Before installing the dependencies listed in the `requirements.txt` file, there are some specific steps to address challenges with installing `onnxruntime`.

### For MacOS Users:
1. Install the `onnxruntime` dependency for `chromadb` using Conda:
    ```bash
    conda install onnxruntime -c conda-forge
    ```


### For Windows Users:
1. Install the Microsoft C++ Build Tools by following this [guide](https://github.com/bycloudai/InstallVSBuildToolsWindows?tab=readme-ov-file).
2. Ensure you complete all steps, including setting the environment variable path.

### Install Remaining Dependencies:
Once the above steps are complete, install the dependencies from the `requirements.txt` file:
```bash
pip install -r requirements.txt
```

### Install Markdown Dependencies:
To handle markdown files, install the following:
```bash
pip install "unstructured[md]"
```

## Step 2: Create the Database

The project uses Chroma DB to store and retrieve data. To create the database, run:
```bash
python create_database.py
```

This script processes your documents and stores them in a vector database for efficient querying.

## Step 3: Query the Database

To query the database, use the following command:
```bash
python query_data.py "How does Alice meet the Mad Hatter?"
```

This will return relevant information from your documents based on the query. 

> **Note:** You need to set up an OpenAI account and configure your OpenAI API key as an environment variable for this step to work.


By following these steps, you can replicate this project and build your own chatbot powered by Langchain and RAG. Happy coding!
