This is the LangChain Practice App.

在这里我开发了三个应用，首先是一个领英信息查找与提取应用**ice_breaker.py**。主要是基于LangChain的LLMChain，大模型使用的是gpt-3.5-turbo。流程是这样的，首先我开发了一个ReAct Agent来完成从网络上检索某个人物领英页面链接的功能，我使用的**AgentType是ZERO_SHOT_REACT_DESCRIPTION:** ZeroShotAgent。 这个代理使用ReAct框架仅根据工具的描述来确定要使用的工具。可以提供任意数量的工具。我提供的工具是一个用来检索领英链接的函数，这里面使用了SerpAPIWrapper，这是langchain内置的一个Google Search API，可以输入一个搜索内容返回相应的搜索结果，结果的形式是一个字典，包含"organic_results"，"answer_box"等字段，可以通过`_process_response` 方法自定义一个CustomSerpAPIWrapper来实现对返回结果的定制化处理。比如我的一个重要修改就是让我的这个api返回res["organic_results"][0]["link"]这个字段，而不是所有的内容。

我也开发了另一个方法接收领英界面链接来爬取领英内容，这个方法调用了一个proxycurl服务API，这是一个提供领英页面抓取和解析的服务，给出领英界面url即可。返回的是一个包含name，self_introduction等内容的Json字符串。我称之为linkedIn_Data，将这个Data放入到我的Langchain Prompt template输入里。辅以format_instruction。对于format_instruction，我使用的是pydantic的自定义Model和langchain的PydanticOutputParser。Pydantic 是一个 Python 库，用于数据验证和设置管理。它主要基于 Python 类型提示来定义数据模型，这使得代码既简洁又易于理解。

第二个项目是一个博客检索应用**query-remote-vector-db**，在这个应用里我使用LangChain的`TextLoader` 把一篇博客的内容txt文件加载到应用里，同时使用`text_splitter`词对文本进行分块，目的是方便使用`OpenAIEmbeddings`工具将文本内容转化为相应分块数量的词嵌入（Word Embedding）向量数据段，并把这些数据保存到`Pinecone` 向量数据库。通过将文档和它们的嵌入向量存储在Pinecone索引中，我们可以在代码里通过LangChain Pinecone类的`from_documents`方法调用向量数据库搜索器retriever，能够高效地执行语义相似度搜索，找到与查询最相关的文档字段。

将这个检索器传入LangChain**`RetrievalQA`** ，这是一个用于检索式问答（Retrieval-based QA）的组件，它结合了语言模型（如OpenAI提供的LLM）和一个检索器（在这个例子中是 **`docsearch.as_retriever()`**）来回答问题。其工作流程通常包括以下步骤：
1. **检索阶段**：使用 **`docsearch`** 从一个或多个文档中检索与问题最相关的信息。这是通过将问题编码为一个向量，然后在向量数据库中查找最近似的文档向量来完成的。
2. **答案生成阶段**：将检索到的文档或段落以及问题本身传递给一个语言模型（如 **`OpenAI()`**），由该模型生成答案。
3. **向量数据库的作用：**在这个系统中，向量数据库起到了至关重要的作用。它允许系统快速有效地从大规模文档集合中检索出与问题最相关的信息。这种基于向量的检索方式比传统的关键词搜索更加灵活和精确，因为它能够捕捉到词义上的相似性，而不仅仅是表面的文本匹配。此外，向量数据库的使用还能大大提高检索速度，这对于实时问答系统来说是非常重要的。

最后的一个项目是一个论文问答应用**pdf-localvectordb-digesting&query**。这个应用使用的是langchain `PyPDFLoader`来加载PDF文件。然后同样地使用`text_splitter`对文件内容进行分块。这次不同的是我是使用的LangChain FAISS向量数据库类进行embedding生成同时把embedding数据保存在了本地而非远程数据库。最后也是使用问答组件构建了一个机器人可以回答用户针对PDF内容提出的问题。
