# safetensor大模型导入ollama
[参考资料-CSDN](https://blog.csdn.net/m0_73365120/article/details/141901884)
## 准备gguf模型文件
下载llama.cpp并完成编译（新版可能不用编译，没试）
```
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make -j # 注意，这里我并没有增加 gpu 的支持，因为这次我只用 llama.cpp 进行模型转化和量化，不用它做推理
```
进入llama.cpp，执行requirments文件，注意把子requirments中涉及torch的删点，否则会卸掉gpu版本重装cpu版本

进入llama.cpp，执行如下命令，将safetensor--->gguf
```
python convert_hf_to_gguf.py --outfile <要导出的文件地址.gguf> <微调后的模型来源目录>
```
## 将gguf导入ollama模型库
创建makefile，格式如下。注意！temperature和ctx_num参数分别定义了模型活跃度和输出长度，越低模型胡咧咧的概率越低。SYSTEM后是默认prompts，需仔细设置
```
FROM PATH_TO_YOUR__GGUF_MODEL
# set the temperature to 1 [higher is more creative, lower is more coherent]
PARAMETER temperature 1
# set the system message
SYSTEM You are Mario from Super Mario Bros. Answer as Mario, the assistant, only.
```
执行如下命令，完成添加
```
ollama create NAME -f ./Modelfile

# NAME: 在ollama中显⽰的名称
# ./Modelfile: 绝对或者相对路径
```


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
- RAG+大模型后胡言乱语的原因：1）所用大模型不够“重”，2）safetensor转的大模型，没有约束好num_ctx和temperature
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
embedding失败的常见原因，用中文embedding模型处理英文知识库或者反之
```
Settings.embed_model = OllamaEmbedding(model_name="mxbai-embed-large")
Settings.embed_model = OllamaEmbedding(model_name="shaw/dmeta-embedding-zh")
```
