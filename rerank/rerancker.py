import os
import numpy as np
# os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
# os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'

from FlagEmbedding import FlagReranker
from langchain_community.vectorstores import FAISS
from embdding_model.embedding import CustomModelScopeEmbeddings

# model_path = r'/home/disks/sdd/hxr/hxr/python/ZTE_challenge/hugging_face/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181'
# reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True) # Setting use_fp16 to True speeds up computation with a slight performance degradation

def chunks_rerancker(query,  retrieve_results, retrieve_k, model_name_or_path = 'BAAI/bge-reranker-v2-m3' ):
    # query_vector = embeddings.embed_query(query).reshape(-1)
    # query_vector = np.array(query_vector).reshape(-1)
    # retrieve_results = db.similarity_search_by_vector(query_vector, k=research_k)
    reranker = FlagReranker(model_name_or_path, use_fp16 =  True)
    scores = reranker.compute_score([[query, result.page_content] for result in retrieve_results], normalize= True)
    # scores 为每一个retrieve_results 对应的分数值
    # scores 和 retrieve_results 都是列表类型
    # 按照 scores从大到小的顺序，重新排列 retrieve_results
    reranged_retrieve_results = [result for _, result in sorted(zip(scores, retrieve_results), reverse=True)]
    return reranged_retrieve_results[:retrieve_k]

# query = '中兴在大模型应用上做了很多尝试'
# db_path = r'D:\python\ZTE_challenge\data_test\db\all_embedding.faiss'
# embeddings = CustomModelScopeEmbeddings()
# db = FAISS.load_local(db_path, embeddings=embeddings, allow_dangerous_deserialization=True)
# scores = chunks_rerancker(query,db, embeddings, retrive_k=100, refine_return_k= 3)





# reranker = FlagReranker(model_path, use_fp16=True)
#
# # 第一个参数是 query, 第二个参数是  passage
# # 可以考虑使用多个 embeding进行检索，然后使用
# score = reranker.compute_score(['query', 'passage'])
# print(score) # -5.65234375 [0.45751953125, 0.04058837890625]
#
# # You can map the scores into 0-1 by set "normalize=True", which will apply sigmoid function to the score
# score = reranker.compute_score(['query', 'passage'], normalize=True)
# print(score) # 0.003497010252573502 [0.6124255753051662, 0.5101457019150742]
#
# scores = reranker.compute_score([['what is panda?', 'hi'], ['what is panda?', 'The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.']])
# print(scores) # [-8.1875, 5.26171875] [-0.196533203125, 0.32373046875, -0.41064453125, -0.08441162109375]
#
# # You can map the scores into 0-1 by set "normalize=True", which will apply sigmoid function to the score
# scores = reranker.compute_score([['what is panda?', 'hi'], ['what is panda?', 'The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.']], normalize=True)
#
# # reranker.compute_score([['what is panda?','a','b','c'], ['what is panda?', 'panda is a cat-like animal', 'panda like eate banboo','I like banboo']], normalize=True)
# print(scores) # [0.00027803096387751553, 0.9948403768236574] [0.45102423978150774, 0.5802331259648752, 0.39875758472286527, 0.4789096162205436]
#
#
# # 功能：计算两条消息之间的相似度,然后返回相似度分数
# #