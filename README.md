# Cardiovascular RAG App

The objective of this project is to create a French cardiovascular expert chatbot that can be used by patients suffering from cardiovascular diseases to provide answers to their health issues. This is a RAG implementation using open source stack. The LLM used for the chatbot is [BioMistral](https://huggingface.co/BioMistral/BioMistral-7B). BioMistral is an opensource LLM finetuned for medical domains. In order to run the application locally on CPU, a [quantized model](https://huggingface.co/MaziyarPanahi/BioMistral-7B-GGUF) of the LLM is used. [PubMedBert](https://huggingface.co/NeuML/pubmedbert-base-embeddings) is the embedding model used for the application. This model is finetuned using sentence-transormers. It outperforms all other sentence transformer models for tasks on medical domain. It creates a 768 dimensional dense vector for embedding. [Qdrant](https://qdrant.tech/) is the vector database used for storing the vectors. Qudrant is a self hosted open source vector database. LangChain and Llama CPP are used as the orchestration framework.


## Data
The `data` folder contains the data used for creating the vectors. You can use any number of files. The data in these files will be be converted to vectors and stored in Qdrant. 


## Architecture

The initial step is to create a vector embedding of all the documents. This is done using the script `ingest.py`. When the user inputs a query, the vectors with the highest similarity are retrieved from the vector database. This is goven as the context for the LLM. THis would help to reduce the context length of the input. In order to facilitate memory, we are using ConversationalRetrivalChain. The previous query and output are passed to the LLM so that it can rewrite the new query. This gives more context to the LLM, and gives better responses. 

## How to Run

<b>Step 1:Setup virtual environment</b>

In order to run the application, you have to create a new Python virtual environment.

```
python3 -m venv env_chatbot
source env_chatbot/bin/activate
```

<b>Step 2: Install all the requirements</b>

```
pip install -r requirements.txt
```

<b>Step 3: Setup Qdrant</b>

Install docker if not done

```
docker pull qdrant/qdrant

docker run -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage:z \
    qdrant/qdrant

```

<b>Step4: Download the quantized version of LLM</b>

Downaload the quantized version of BioMistral BioMistral-7B.Q4_K_M.gguf from [here](https://huggingface.co/MaziyarPanahi/BioMistral-7B-GGUF) to the project directory.

<b>Step5: Ingest data to Qdrant collection</b>

```
python3 ingest.py
```

<b>Step6: To test the RAG pipeline without deploying the API</b>

```
python3 test_rag_app.py
```

<b>Step7: Start the FastAPI server </b>

```
uvicorn main:app
```





