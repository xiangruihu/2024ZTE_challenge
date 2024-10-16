import os
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
# os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
# os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'

from langchain_community.embeddings import  ModelScopeEmbeddings



def save_chunk_embeding(chunks,embedding_model, save_path):
    db = FAISS.from_documents(chunks, embedding_model)
    db.save_local(save_path)
    print('saving chunk embeddings to {}'.format(save_path))


def merge_all_chunk_embedding(chunks_embed_dir, embeddings, save_name='all_embedding.faiss'):
    # 获取目录中的所有 FAISS 索引文件
    faiss_files = [f for f in os.listdir(chunks_embed_dir) if f.endswith('.faiss')]

    if not faiss_files:
        raise ValueError("No FAISS files found in the specified directory.")

    # 初始化合并索引的变量
    db_merged = None

    # 逐个加载和合并 FAISS 索引
    for i, faiss_file in enumerate(faiss_files):
        file_path = os.path.join(chunks_embed_dir, faiss_file)
        if db_merged is None:
            db_merged = FAISS.load_local(file_path, embeddings=embeddings, allow_dangerous_deserialization=True)
        else:
            db_temp = FAISS.load_local(file_path, embeddings=embeddings, allow_dangerous_deserialization=True)
            db_merged.merge_from(db_temp)

    # 保存合并后的索引到指定文件
    save_path = os.path.join(chunks_embed_dir, save_name)
    db_merged.save_local(save_path)

    print(f"FAISS indices merged and saved successfully to {save_path}.")


# db_dir = r'D:\python\ZTE_challenge\data\embedding_db'
# embeddings = CustomModelScopeEmbeddings()
# merge_all_chunk_embedding(db_dir,embeddings=embeddings)



#
# # def save_chunks(chunks,embedding_model):
#
#
# embedding_model = CustomModelScopeEmbeddings("BAAI/bge-m3")
# text = "大家好，这里是来自电子科技大学的团队，我们想要夺得冠军！"
# text = '嘿哈加好，我们好好学习，天天向上，希望能够拿到一个好的名次，希望能够进入决赛，加油！'
# text_splitter = RecursiveCharacterTextSplitter(chunk_size = 6, chunk_overlap=2)
# chunk = text_splitter.create_documents([text])
#
# # db = FAISS.from_documents(chunk, embedding_model)
# # db.save_local('LLM2.faiss')
# # # qurry = '科技大学'
# # #
# # print('d')
#
# save_chunk_embeding(chunk, CustomModelScopeEmbeddings("BAAI/bge-m3"), 'LLM_test.faiss')