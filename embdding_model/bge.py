# import os
# os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
# os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
from FlagEmbedding import BGEM3FlagModel

from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import ModelScopeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
model = SentenceTransformer("BAAI/bge-m3")

model = BGEM3FlagModel('BAAI/bge-m3',
                       use_fp16=True) # Setting use_fp16 to True speeds up computation with a slight performance degradation

# sentences_1 = ["What is BGE M3?", "Defination of BM25"]
# sentences_2 = ["BGE M3 is an embedding model supporting dense retrieval, lexical matching and multi-vector interaction.",
#                "BM25 is a bag-of-words retrieval function that ranks a set of documents based on the query terms appearing in each document"]
#
# embeddings_1 = model.encode(sentences_1,
#                             batch_size=12,
#                             max_length=8192, # If you don't need such a long length, you can set a smaller value to speed up the encoding process.
#                             )['dense_vecs']
# embeddings_2 = model.encode(sentences_2)['dense_vecs']
#
#
# similarity = embeddings_1 @ embeddings_2.T
# print(similarity)   #  相关性检索

# [[0.6265, 0.3477], [0.3499, 0.678 ]]


def get_embedding(sentence, model_name= 'bge-3'):
    if model_name == 'bge-3':
        model = BGEM3FlagModel('BAAI/bge-m3',use_fp16=True)

    else:
        model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

    embeddings = model.encode(sentence,
                                batch_size=12,
                                max_length=8192,
                                # If you don't need such a long length, you can set a smaller value to speed up the encoding process.
                                )['dense_vecs']

    return embeddings

# sentenses = ['中国是世界上人口最多的国家之一',
#              '中国人口非常多',
#              '小明喜欢吃苹果',
#              '适度睡眠有益于健康']
#
# embedding = get_embedding(sentenses)
#
# print('d')



from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import ModelScopeEmbeddings

# 初始化 SentenceTransformer 模型
model = SentenceTransformer("BAAI/bge-m3")

# 定义一个嵌入函数来调用 SentenceTransformer 模型
# def embedding_function(texts):
#     return model.encode(texts, convert_to_tensor=True)
#
# # 使用 ModelScopeEmbeddings 包装你的嵌入函数
# # model_scope_embeddings = ModelScopeEmbeddings(embedding_function)
#
# # 示例文本
# texts = ["Hello, world!", "This is a test."]
#
# # 获取嵌入向量
# embeddings = model_scope_embeddings.embed(texts)
#
# print(embeddings)

class CustomModelScopeEmbeddings:
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)
    def embed_documents(self, texts):
        return self.model.encode(texts, convert_to_tensor=True)

# text = "大家好，这里是来自电子科技大学的团队，我们想要夺得冠军！"
# text_splitter = RecursiveCharacterTextSplitter(chunk_size = 6, chunk_overlap=2)
# txts = text_splitter.create_documents([text])
#
# embedding_model = CustomModelScopeEmbeddings("BAAI/bge-m3")
# chunk = ['ddd',
#          'ddddaa']
#
# vector_db = FAISS.from_documents(chunk, embedding_model)