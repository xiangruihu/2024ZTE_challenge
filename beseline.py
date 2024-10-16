import os
import numpy as np
from tqdm import tqdm
import dashscope
from langchain_community.vectorstores import FAISS
from dataloader.pdf_loader import get_all_document, get_chunks, get_all_chunks
# from embdding_model.utils import get_embedding_base, get_embedding_base_merge
from embdding_model.embedding import CustomModelScopeEmbeddings
from build_database.build_database import merge_all_chunk_embedding, save_chunk_embeding
from utili import load_questions,extract_answer_and_reason, save_result, save_result_explain, extract_keywords
from qwen_model.qwen import build_qury_enhancer_template, build_prompt_template,build_three_judge_prompt_template,build_finnally_judgement,build_judgement_extract
from qwen_model.qwen import get_completion
from post_process.final_judgement import retrieve_and_process_answers
# from rerank.rerancker import chunks_rerancker


# print()
if __name__ == '__main__':

    # cn_data_root = 'clean_data/cn'
    # db_save_dir = 'data/embedding_db'

    # cn_data_root = 'data_test/document'
    # db_save_dir = 'data_test/db'

    cn_data_root = 'clean_data/cn_en'
    # db_save_dir = 'clean_data/db_qwen2_7b'
    # db_save_dir = 'clean_data/db_mul_e5_l'
    db_save_dir = 'clean_data/db'

    all_db_name = 'all_embedding.faiss'
    question_file_path = 'test_B.csv'
    result_path = 'result_csvs/test_result_B.csv'
    result_explain_path = os.path.join(result_path.split('/')[0],result_path.split('/')[1][:-4] + '_explain.csv' )
    embdding_model_path = "BAAI/bge-m3"
    # embdding_model_path = 'intfloat/multilingual-e5-large-instruct' # 上传值 hxr
    # embdding_model_path = 'Alibaba-NLP/gte-Qwen2-1.5B-instruct'
    # embdding_model_path = '/home/disks/sdd/hxr/hxr/python/ZTE_challenge/hugging_face/gte_qwen2_1_5B'
    # embdding_model_path = '/home/disks/sdd/hxr/hxr/python/ZTE_challenge/hugging_face/multilingual-e5-large-instruct'
    # embdding_model_path = "Alibaba-NLP/gte-Qwen2-7B-instruct"
    # embdding_model_path = '/home/disks/sdd/hxr/hxr/python/ZTE_challenge/hugging_face/gte-Qwen2-7B-instruct'
    # embdding_model_path = r'hugging_face/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181'
    # embdding_model_path = r'/home/disks/sdd/hxr/hxr/python/ZTE_challenge/hugging_face/jina'
    # embdding_model_path = r'/home/disks/sdd/hxr/hxr/python/ZTE_challenge/hugging_face/multilingual-e5-large-instruct'
    model_name_2 = 'qwen1.5-14b-chat'
    model_name_3 = 'qwen1.5-14b-chat' # 会输出input token 的数量
    model_name = 'qwen1.5-14b-chat'
    # final_judge_model_name = 'qwen2-72b-instruct'
    final_judge_model_name = 'qwen1.5-14b-chat'
    query_enhancer_model_name = 'qwen-14b-chat'

    # embedding_name = "BAAI/bge-m3"
    # result_path
    cn_chunk_size = 160
    cn_chunk_overlap = 40
    retrive_k = 6
    # research_k  = 6
    max_try = 6

    questions = load_questions(question_file_path)
    all_db_path = os.path.join(db_save_dir, all_db_name)

    embeddings = CustomModelScopeEmbeddings(embdding_model_path)
    if os.path.exists(all_db_path):
        pass
    else:
        cn_data_base = get_all_document(cn_data_root)
        # 使用一个大的列表进行 装载chunks
        ch_chunks = get_all_chunks(cn_data_base, chunk_size = cn_chunk_size, overlap_size= cn_chunk_overlap)
        # 获取embedding
    

    if os.path.exists(all_db_path):
        db = FAISS.load_local(all_db_path, embeddings=embeddings, allow_dangerous_deserialization=True)
    else:

        for index, key in enumerate(ch_chunks.keys()):
            if os.path.exists(os.path.join(db_save_dir,key[:-3].replace(' ', '') + 'faiss')):
                print('the file of {} alread exist '.format(key[:-3].replace(' ', '') + 'faiss'))
                continue
            save_chunk_embeding(ch_chunks[key],embeddings,
                                save_path=os.path.join(db_save_dir,key[:-3].replace(' ', '') + 'faiss'))

        merge_all_chunk_embedding(db_save_dir,embeddings= embeddings)
        db = FAISS.load_local(all_db_path, embeddings=embeddings, allow_dangerous_deserialization=True)

    # 一次只能传入一个querry
    # query = '江苏移动和中兴公司达成合作'
    prompt_template = build_prompt_template()
    # prompt_template = build_en_prompt_template()
    # prompt_template = build_three_judge_prompt_template()
    qury_extract_template = build_qury_enhancer_template()
    final_judge_template = build_finnally_judgement()
    # ans_extract_template = build_judgement_extract()
    # query = 'JiangSu Provience have been establish a lot of cooporation with ZTE cooporation'
    # query_vector = embeddings.embed_query(query).reshape(-1)
    # query_vector = np.array(query_vector).reshape(-1)
    # retrieve_results = db.similarity_search_by_vector(query_vector,k= retrive_k)
    #
    #
    # custom_message = prompt_template.format_messages(
    #     documents=retrieve_results,
    #     # question = '中国的人口数量非常多',
    #     question=query,
    # )
    #
    # print(get_completion(custom_message[0].content))

    for id, que in tqdm(questions.items()):
        query = que
        query_vector = embeddings.embed_query(query).reshape(-1)
        query_vector = np.array(query_vector).reshape(-1)
        retrieve_results = db.similarity_search_by_vector(query_vector, k=retrive_k)
        # 对检索结果进行重排
        # rerank_result = retrieve_results()
        # rerank_result = chunks_rerancker(que,retrieve_results,retrieve_k=retrive_k )

        custom_message = prompt_template.format_messages(
            documents=retrieve_results,
            # documents = rerank_result,
            question=query,
        )

        try_time = 0
        while try_time < max_try:
            try:
                # 连接失败
                ans_list = []
                response = get_completion(custom_message[0].content, model_name )
                ans, reson = extract_answer_and_reason(response)
                # ans, reson =
                ans_list.append(ans)



                save_result(result_path, id, ans)
                save_result_explain(result_explain_path, id, ans, reson)
                print('the answer fo question {} is {}'.format((int(id)),ans))
                break

            except Exception as e:
                try_time += 1
                print(f"Error processing question {id}, retry {try_time}/{max_try}: {e}")
                if try_time == max_try:
                    print(f"Failed to process question {id} after {max_try} retries.")
