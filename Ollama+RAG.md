# 传统知识库（RAG）：
## 如何分割文档？
- 暴力分割，直接扔进去按照行数、条目数量分割
- 按照规则分割：人工清洗->脚本处理
- 内容+摘要两步走（需要自己写embedding脚本，先根据摘要决定要不要继续embedding内容）
## 知识库的性能
- 速度
- 准确度
## RAG的效果不好
- 数据处理
- Prompt
--------------------- 
## 使用ollama读取知识库并进行embedding

``` python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import StorageContext, load_index_from_storage

#加载数据
documents = SimpleDirectoryReader("/root/autodl-tmp/medicine/data_rag").load_data()

#Emb
Settings.embed_model = OllamaEmbedding(model_name="mxbai-embed-large")

#Post processing
Settings.llm = Ollama(model="qwen2:1.5b", request_timeout=360)


index = VectorStoreIndex.from_documents(
            documents
            )
# 将embedding结果存到指定位置，以后可以复用
index.storage_context.persist(persist_dir="/root/autodltmp/medicine/embedding")
query_engine = index.as_query_engine()
# 执行查询并获取响应
print("qwen2-1.5b_chat--------------按e结束--------------")
while True:
    ask = input("输入对话：")
    response = query_engine.query(ask)
    #response = query_engine.query("")
    print(f'大模型回答：{response}')
```
读取上次embedding的结果:
```
storage_context = StorageContext.from_defaults(persist_dir="/root/law/embedding")
index = load_index_from_storage(storage_context)
```
