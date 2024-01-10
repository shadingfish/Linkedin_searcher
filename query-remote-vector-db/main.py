from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_community.vectorstores import Pinecone
from langchain.chains import RetrievalQA
import os
import pinecone

print("Initializing vector...")
pinecone.init(api_key=os.environ.get("PINECONE_API_KEY"), environment="gcp-starter")

if __name__ == "__main__":
    print("Hello VectorStore!")
    script_dir = os.path.dirname(__file__)  # 脚本所在目录
    file_path = os.path.join(script_dir, "mediumblogs", "mediumblog1.txt")
    loader = TextLoader(file_path, encoding="utf-8")
    document = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)

    print(len(texts))

    print("Create embeddings...")
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
    print("Saving vectors...")
    docsearch = Pinecone.from_documents(
        texts, embeddings, index_name="medium-blogs-embeddings-index"
    )

    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(),
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        return_source_documents=True,
    )
    query = "What is a vector DB? Give me a 15 word answer for a beginner"
    result = qa({"query": query})
    print(result)
