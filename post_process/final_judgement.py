from qwen_model.qwen import get_completion,build_judgement_extract
from utili import extract_answer_and_reason

def retrieve_and_process_answers(db, query_vector, retrive_k, query, prompt_template, model_name, model_name_2,
                                 model_name_3, final_judge_template, final_judge_model_name, retrieve_piece):
    # ans_extract_template = build_judgement_extract()
    ans_l = []
    # Step 1: Retrieve results 2*k
    retrieve_results_shift = db.similarity_search_by_vector(query_vector, k=int((retrieve_piece ) * retrive_k))
    retrieve_results_shift = retrieve_results_shift[int(retrive_k * (retrieve_piece - 1)):]

    # Step 2: Shift message
    shift_message = prompt_template.format_messages(
        documents=retrieve_results_shift,
        question=query,
    )

    # Step 3: Get first model response
    response = get_completion(shift_message[0].content, model_name)
    ans, reson = extract_answer_and_reason(response)
    ans_l.append(ans)
    print('get ans1 {}'.format(ans))

    # Step 4: Get second model response
    response_2 = get_completion(shift_message[0].content, model_name_2)
    ans_2, reson_2 = extract_answer_and_reason(response_2)
    ans_l.append(ans_2)
    print('get ans2 {}'.format(ans_2))

    # Step 5: Get third model response
    response_3 = get_completion(shift_message[0].content, model_name_3)
    ans_3, reson_3 = extract_answer_and_reason(response_3)
    ans_l.append(ans_3)
    print('get ans3 {}'.format(ans_3))

    # Step 6: Construct final judge message
    final_judge_message = final_judge_template.format_messages(
        query=query,
        resons='答案和解释1：' + reson + '\\n\n' + '答案和解释2:  ' + reson_2 + '\\n\n' + '答案和解释3:  ' + reson_3
    )

    # Step 7: Get final response
    finaly_response = get_completion(final_judge_message[0].content, final_judge_model_name)
    # extract_judge_message = ans_extract_template.format_messages(reson=finaly_response)
    # finaly_response = get_completion(extract_judge_message[0].content,final_judge_model_name )
    final_ans, final_reson = extract_answer_and_reason(finaly_response)
    # ans_l.append(final_ans)
    print('get final ans {}'.format(final_ans))

    return final_ans, final_reson, ans_l
