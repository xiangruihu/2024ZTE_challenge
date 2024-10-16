from sentence_transformers import SentenceTransformer
import torch
import os
# os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
# os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
class CustomModelScopeEmbeddings:
    def __init__(self, model_name = "BAAI/bge-m3" ):
        self.model_name = model_name
        self.model = SentenceTransformer(self.model_name)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    def embed_documents(self, texts):
        return self.model.encode(texts,  batch_size=64, device= self.device, convert_to_tensor=True).cpu()
    def embedding_function(self, text):
        return self.model.encode(text, convert_to_tensor=True, device=self.device).cpu()
    def embed_query(self, txt:str  ):
        return self.model.encode(txt, convert_to_tensor=True, device=self.device).cpu()
        # 添加 __call__ 方法
    def __call__(self, text):
        return self.embedding_function(text)

#  384 维度
# model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
# 1.5B  1536 dimention
#  model = SentenceTransformer("Alibaba-NLP/gte-Qwen2-1.5B-instruct", trust_remote_code=True)

# model = SentenceTransformer("Alibaba-NLP/gte-Qwen2-7B-instruct", trust_remote_code=True)

# # embeddings = CustomModelScopeEmbeddings("Alibaba-NLP/gte-Qwen2-7B-instruct")  # donw_loaded
# embeddings = CustomModelScopeEmbeddings("Alibaba-NLP/gte-Qwen2-1.5B-instruct")
# # embeddings = CustomModelScopeEmbeddings('sentence-transformers/all-MiniLM-L6-v2')
# # print
# # query = ['你好']
# embed = embeddings.embed_query('你好')
# print(embed)
