from langchain_community.vectorstores import FAISS
from db2 import CustomModelScopeEmbeddings
import faiss

# 初始化嵌入模型
embeddings = CustomModelScopeEmbeddings("BAAI/bge-m3")

# 加载多个 FAISS 索引
vector_db1 = FAISS.load_local('LLM.faiss', embeddings=embeddings, allow_dangerous_deserialization=True)
vector_db2 = FAISS.load_local('LLM2.faiss', embeddings=embeddings, allow_dangerous_deserialization=True)
# 可以继续加载更多的索引...

vector_db2.merge_from(vector_db1)

print("D")
