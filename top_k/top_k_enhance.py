import numpy as np
from tqdm import tqdm
from langchain_community.vectorstores import FAISS
from embdding_model.embedding import CustomModelScopeEmbeddings
from utili import load_questions
if __name__ == '__main__':
    topk = 6
    lap = 40
    db_path = r'../clean_data/db/all_embedding.faiss'
    question_path = r'../test_A.csv'
    save_top_path = r'top_k_research.txt'
    embedding_path = "BAAI/bge-m3"

    embeddings = CustomModelScopeEmbeddings(embedding_path)
    db = FAISS.load_local(db_path, embeddings=embeddings, allow_dangerous_deserialization=True)
    questions = load_questions(question_path)

    for id, question in tqdm(questions.items()):
        query_vector = embeddings.embed_query(question).reshape(-1)
        query_vector = np.array(query_vector).reshape(-1)
        retrieve_results = db.similarity_search_by_vector(query_vector, k = topk)

        # 保存 retrieve_results 和问题
        with open(save_top_path, 'a',encoding='utf-8') as f:
            f.write(str(id) + '.' + question)
            f.write('\n')
            for result in retrieve_results:
                f.write(str(id) + '.' + str(result))
                f.write('\n')

