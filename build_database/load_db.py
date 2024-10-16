from langchain_community.vectorstores import FAISS
from db2 import CustomModelScopeEmbeddings

embeddings = CustomModelScopeEmbeddings("BAAI/bge-m3")
vector_db = FAISS.load_local('LLM.faiss',embeddings=embeddings, allow_dangerous_deserialization=True)

retriever = vector_db.as_retriever(search_kwargs = {'k':5})

query = '科技大学'

# 使用检索器来检索相似文档
results = retriever.retrieve(query)

# 打印结果
for i, result in enumerate(results):
    print(f"Result {i + 1}:")
    print(f"Document: {result['text']}")
    print(f"Score: {result['score']}")

print('d')