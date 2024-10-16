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
from qwen_model.qwen import build_qury_enhancer_template, build_three_judge_prompt_template,build_finnally_judgement,build_judgement_extract
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
    question_file_path = 'test_A.csv'
    result_path = 'result_csvs/multi_model_better_extract_k6_s_160_ol_40_m_multi.csv'
    result_explain_path = os.path.join(result_path.split('/')[0],result_path.split('/')[1][:-4] + '_explain.csv' )
    embdding_model_path = "BAAI/bge-m3"
    # embdding_model_path = "Alibaba-NLP/gte-Qwen2-7B-instruct"
    # embdding_model_path = '/home/disks/sdd/hxr/hxr/python/ZTE_challenge/hugging_face/gte-Qwen2-7B-instruct'
    # embdding_model_path = r'hugging_face/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181'
    # embdding_model_path = r'/home/disks/sdd/hxr/hxr/python/ZTE_challenge/hugging_face/jina'
    # embdding_model_path = r'/home/disks/sdd/hxr/hxr/python/ZTE_challenge/hugging_face/multilingual-e5-large-instruct'
    model_name_2 = 'qwen1.5-14b-chat'
    model_name_3 = 'qwen-14b-chat' # 会输出input token 的数量
    model_name = 'qwen2-7b-instruct'
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
    # prompt_template = build_prompt_template()
    # prompt_template = build_en_prompt_template()
    prompt_template = build_three_judge_prompt_template()
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

        query_extract_message = qury_extract_template.format_messages(
            query = query,
        )

        try_time = 0
        while try_time < max_try:
            try:
                # 连接失败
                ans_list = []
                response = get_completion(custom_message[0].content, model_name )
                ans, reson = extract_answer_and_reason(response)
                ans_list.append(ans)
                # print('the ans of piece1 ans1 is {}'.format(ans))

                response_2 = get_completion(custom_message[0].content, model_name_2)
                ans_2,reson_2 = extract_answer_and_reason(response_2)
                ans_list.append(ans_2)
                # print('the ans of piece1 ans2 is {}'.format(ans_2))


                response_3 = get_completion(custom_message[0].content, model_name_3)
                ans_3, reson_3 = extract_answer_and_reason(response_3)
                ans_list.append(ans_3)
                # print('the ans of piece1 ans3 is {}'.format(ans_3))
                print(' the ans of question {} are {} {} {}'.format(id, ans, ans_2, ans_3))
                if not (ans == ans_2 == ans_3) or 'N' in (ans, ans_2, ans_3):


                    final_judge_message = final_judge_template.format_messages(
                        query=query,
                        resons='答案和解释1：' + reson + '\\n\n' + '答案和解释2:  ' + reson_2 + '\\n\n' + '答案和解释3:  ' + reson_3
                    )

                    finaly_response = get_completion(final_judge_message[0].content, final_judge_model_name)

                    # extract_judge_message = ans_extract_template.format_messages(reson = finaly_response)
                    # finaly_response = get_completion(extract_judge_message[0].content, final_judge_model_name)

                    final_ans_piece1, final_reson_piece1 = extract_answer_and_reason(finaly_response)

                    print('final reson of piece1 is {}'.format(final_ans_piece1))


                    if final_ans_piece1 == 'N':

                    # if not (ans == ans_2 == ans_3) or 'N' in (ans, ans_2, ans_3):



                        # try_time = 1
                        retrieve_piece = 2
                        final_ans_piece2, final_reson_piece2,final_ans_l_piece2 = retrieve_and_process_answers(db, query_vector, retrive_k, query, prompt_template,model_name,model_name_2,
                                                     model_name_3,final_judge_template,final_judge_model_name, retrieve_piece)
                        ans_list +=final_ans_l_piece2
                        print('the final_ans of retrieve  piece 2 is {}'.format(final_ans_piece2))

                        if final_ans_piece2 == 'N':
                            # 继续处理一轮
                            retrieve_piece = 3
                            final_ans_piece3, final_reson_piece3, final_ans_l_piece3 = retrieve_and_process_answers(db, query_vector, retrive_k,
                                                                                               query, prompt_template,
                                                                                               model_name, model_name_2,
                                                                                               model_name_3,
                                                                                               final_judge_template,
                                                                                               final_judge_model_name,
                                                                                               retrieve_piece)
                            ans_list += final_ans_l_piece3
                            print('the final_ans of retrieve piece 3 is {}'.format(final_ans_piece3))

                            if final_ans_piece3 == 'N':
                                # 做最后决断 # 修改query 策略
                                # query_enhanced = get_completion(query_extract_message[0].content, query_enhancer_model_name)
                                # key_words  = extract_keywords(query_enhanced)
                                # a =0
                                # 继续嵌套
                                #
                                count_F = ans_list.count('F')
                                count_N = ans_list.count('N')
                                count_T = ans_list.count('T')
                                ans = 'T' if count_T > count_F else 'F'
                                reson = 'Can not get agreen ans after 3 piece '
                                print(reson)
                            else:
                                ans, reson = final_ans_piece3, final_reson_piece3

                        else:
                            ans, reson = final_ans_piece2, final_reson_piece2

                    else:
                        ans, reson = final_ans_piece1, final_reson_piece1






                save_result(result_path, id, ans)
                save_result_explain(result_explain_path, id, ans, reson)
                print('the answer fo question {} is {}'.format((int(id)),ans))
                break

            except Exception as e:
                try_time += 1
                print(f"Error processing question {id}, retry {try_time}/{max_try}: {e}")
                if try_time == max_try:
                    print(f"Failed to process question {id} after {max_try} retries.")
