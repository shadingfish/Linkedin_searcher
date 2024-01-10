import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

if __name__ == "__main__":
    print("Hello PDF Reader!")
    script_dir = os.path.dirname(__file__)  # 脚本所在目录
    file_path = os.path.join(script_dir, "react_paper.pdf")
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(
        chunk_size=1000, chunk_overlap=30, separator="\n"
    )
    docs = text_splitter.split_documents(documents=documents)

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    save_path = os.path.join(script_dir, "faiss_index_react")
    # vectorstore.save_local(save_path)

    new_vectorstore = FAISS.load_local(save_path, embeddings)

    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(), chain_type="stuff", retriever=new_vectorstore.as_retriever()
    )
    res = qa.run("Give me the gist of RaAct in 3 sentences")
    print(res)
