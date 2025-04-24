
from langchain_community.llms import LlamaCpp
from langchain_community.embeddings import SentenceTransformerEmbeddings
from fastapi import FastAPI, Request, Form, Response
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.encoders import jsonable_encoder
from qdrant_client import QdrantClient
from langchain_community.vectorstores import Qdrant
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import os
import json


templates = Jinja2Templates(directory="templates")
# app.mount("/static", StaticFiles(directory="static"), name="static")

local_llm = "BioMistral-7B.Q4_K_M.gguf"

# Make sure the model path is correct for your system!
llm = LlamaCpp(
    model_path= local_llm,
    temperature=0.7,
    max_tokens=4096,
    top_p=1,
    n_ctx= 2048
)

print("LLM Initialized....")


embeddings = SentenceTransformerEmbeddings(model_name="NeuML/pubmedbert-base-embeddings")

url = "http://localhost:6333"

client = QdrantClient(
    url=url, prefer_grpc=False
)

db = Qdrant(client=client, embeddings=embeddings, collection_name="vector_db")


retriever = db.as_retriever(search_kwargs={"k":1})
system_prompt = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}

Question:
{input}

Only return the helpful answer. Answer must be detailed and well explained.
Helpful answer:
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)


query = "What is breast cancer ?"
question_answer_chain = create_stuff_documents_chain(llm, prompt)
chain = create_retrieval_chain(retriever, question_answer_chain)
docs = retriever.get_relevant_documents(query)
context = "\n\n".join(doc.page_content for doc in docs)
print(prompt.format_messages(context=context, input=query))
response = chain.invoke({"input": query})
answer = response['answer']

source_document = docs[0].page_content if docs else "No source document found"
doc = docs[0].metadata.get('source', 'N/A') if docs else "N/A"

response_data = jsonable_encoder(json.dumps({"answer": answer,"context":context, "source_document": source_document, "doc": doc}))
res = Response(response_data)

print("Réponse :", response)
print("Context", context)

if docs:
    print("Source doc:", source_document)
    print("Source metadata:", doc)
else:
    print("Aucun document trouvé.")
