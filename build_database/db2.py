import os
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.embeddings import ModelScopeEmbeddings

import numpy as np


class CustomModelScopeEmbeddings:
    def __init__(self, model_name):
        super(CustomModelScopeEmbeddings, self).__init__()
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        return self.model.encode(texts)

    def embedding_function(self, text):
        return self.model.encode([text])[0]

    def embed_query(self, txt: str):
        return self.model.encode([txt])[0]

    def _embed_query(self, txt: str):
        return self.model.encode([txt])[0]

    # 添加 __call__ 方法
    def __call__(self, text):
        return self.embedding_function(text)


# 初始化嵌入模型
embedding_model = CustomModelScopeEmbeddings("BAAI/bge-m3")

# 示例文本和文本拆分
text = "大家好，这里是来自电子科技大学的团队，我们想要夺得冠军！"
text_splitter = RecursiveCharacterTextSplitter(chunk_size=6, chunk_overlap=2)
chunks = text_splitter.create_documents([text])

# 使用 FAISS 创建向量存储
db = FAISS.from_documents(chunks, embedding_model)
db.save_local('LLM.faiss')

# 查询示例
query = '科技大学'
query = ['科技', '大学']
query_vector = embedding_model.embed_query(query)

# 确保 query_vector 是一个 2 维数组
# 要求querry 是一个一维的
query_vector = np.array(query_vector).reshape( -1)

# 调用 similarity_search_by_vector 方法
result_simi = db.similarity_search_by_vector(query_vector, k= 4)

# 打印结果
print(result_simi)
