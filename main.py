from utili import load_questions, save_result, save_result_explain, extract_answer_and_reason

from tqdm import tqdm
from dashscope import Application


def call_agent_app(question):
    response = Application.call(app_id='',
                                prompt=question,
                                api_key=api_keys,

                                )
    return response




if __name__ == '__main__':
    api_keys = 'your api key'
    # question_file =  r'G:\竞赛\中兴捧月杯\知识工程\知识工程\test_A.csv'
    question_file = r'test_A.csv'
    result_path = r'aliapi_result3.csv'
    result_explain_path = r'aliapi_result_exp3.csv'


    questions = load_questions(question_file)

    max_try = 6

    for id, que in tqdm(questions.items()):

        # 函数里的request 请求可能会产生报错， 设置一定的容错机制，如果该问题出错了，则重复调用，并打印出相应的错误

        try_time = 0
        while try_time < max_try:
            try:
                response = call_agent_app(que)


                ans, reson = extract_answer_and_reason(response.output.text)

                save_result(result_path, id, ans)
                save_result_explain(result_explain_path, id, ans, reson)
                break

            except Exception as e:
                try_time +=1
                print(f"Error processing question {id}, retry {try_time}/{max_try}: {e}")
                if try_time == max_try:
                    print(f"Failed to process question {id} after {max_try} retries.")

        # print("d")
