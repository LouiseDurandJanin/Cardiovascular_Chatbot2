from langchain import PromptTemplate
from langchain_community.llms import LlamaCpp
from langchain.chains import RetrievalQA
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import SystemMessagePromptTemplate
from langchain_community.embeddings import SentenceTransformerEmbeddings
from fastapi import FastAPI, Request, Form, Response
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.encoders import jsonable_encoder
from qdrant_client import QdrantClient
from langchain_community.vectorstores import Qdrant
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import os
import json


app = FastAPI()

templates = Jinja2Templates(directory="templates")
# app.mount("/static", StaticFiles(directory="static"), name="static")

local_llm = "BioMistral-7B.Q4_K_M.gguf"

# Make sure the model path is correct for your system!
llm = LlamaCpp(
    model_path= local_llm,
    temperature=0.3,
    max_tokens=2048,
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
system_prompt = (
"Use the given context to answer the question. "
"If you don't know the answer, say you don't know. "
"Use three sentence maximum and keep the answer concise. "
"Context: {context}"
)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)
#chat_history = []
#prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/get_response")
async def get_response(query: str = Form(...)):

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    chain = create_retrieval_chain(retriever, question_answer_chain)
    docs = retriever.get_relevant_documents(query)
    context = "\n\n".join(doc.page_content for doc in docs)
    response = chain.invoke({"input": query})
    answer = response['answer']

    source_document = docs[0].page_content if docs else "No source document found"
    doc = docs[0].metadata.get('source', 'N/A') if docs else "N/A"

    response_data = jsonable_encoder(json.dumps({"answer": answer,"context":context, "source_document": source_document, "doc": doc}))
    res = Response(response_data)
    return res
''''
@app.post("/get_response")
async def get_response(query: str = Form(...)):
    # Create the custom chain
    if llm is not None and db is not None:
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm, 
            retriever=retriever, 
            # memory=memory,
            # get_chat_history=chat_history, 
            # return_source_documents=True,
            # combine_docs_chain_kwargs={'prompt': prompt}
        )
    else:
        print("LLM or Vector Database not initialized")

    prompt = PromptTemplate(template=prompt_template, input_variables=["chat_history", 'question'])


    # chain.combine_docs_chain.llm_chain.prompt.messages[0] = SystemMessagePromptTemplate.from_template(prompt)
    response = chain({"question": query, "chat_history": chat_history})
    print(response)
    answer = response['answer']
    chat_history.append((query, answer))
    # source_document = response['source_documents'][0].page_content
    # doc = response['source_documents'][0].metadata['source']
    response_data = jsonable_encoder(json.dumps({"answer": answer}))
    
    res = Response(response_data)
    return res
'''