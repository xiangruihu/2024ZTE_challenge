# import os
# os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
# os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
from FlagEmbedding import BGEM3FlagModel
from langchain_community.vectorstores import FAISS
import faiss
from sentence_transformers import SentenceTransformer

import numpy as np


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

from langchain_community.vectorstores import FAISS

def get_embedding_base(chunks_dict):
    embedding_dict = {}
    sentence_dict = {}
    db_dict = {}
    dbs = []
    # model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
    for key_name in chunks_dict.keys():
        sentens = list(chunks_dict[key_name])
        sentens = [page.page_content for page in sentens]
        sentence_dict[key_name] = sentens

    # for key_name in chunks_dict.keys():
    #     embed1 = get_embedding(sentence_dict[key_name])
    #     txts_embed_pairs = zip(sentence_dict[key_name], embed1)
    #     db = FAISS.from_embeddings(txts_embed_pairs, embed1)
    #     dbs.append(db)

    # 遍历 sentence_dict 得到每个dict 中的db 然后将db进行合并得到一整个db

    embed1 = get_embedding(sentence_dict['jx202405.pdf'])
    txts_embed_pairs = zip(sentence_dict['jx202405.pdf'], embed1)
    db = FAISS.from_embeddings(txts_embed_pairs, embed1)

    return db


def get_embedding_base_merge(chunks_dict):
    sentence_dict = {}
    dbs = []

    for key_name in chunks_dict.keys():
        sentens = list(chunks_dict[key_name])
        sentens = [page.page_content for page in sentens]
        sentence_dict[key_name] = sentens

    for key_name in sentence_dict.keys():
        embed1 = get_embedding(sentence_dict[key_name])
        txts_embed_pairs = zip(sentence_dict[key_name], embed1)
        db = FAISS.from_embeddings(txts_embed_pairs,embed1)
        dbs.append(db)

    # 初始化空的FAISS索引用于合并
    embedding_dim = embed1.shape[1]
    merged_index = faiss.IndexFlatL2(embedding_dim)
    id_index = faiss.IndexIDMap(merged_index)

    current_id = 0
    for db in dbs:
        embeddings = db.index.reconstruct_n(0, db.index.ntotal)
        ids = np.arange(current_id, current_id + embeddings.shape[0])
        id_index.add_with_ids(embeddings, ids)
        current_id += embeddings.shape[0]

    merged_db = FAISS(id_index)

    return merged_db