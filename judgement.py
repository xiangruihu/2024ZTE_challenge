from utili import find_different_answers
import pandas as pd

def get_judge_statement(question_file,explain_file1, explain_file2, diff_index):
    '''
    :param question_file: csv_file # 表头： id, question
    :param explain_file1: csv_file # 表头: id, answer, explain
    :param explain_file2: csv_file # 表头: id, answer, explain
    :param diff_index: list 答案不同的列表 [11,45,49,....]
    :return: id, question, explain_1, explain_2 ( 返回答案不同的问题的id，对应的不同answer，以及解释
    '''
    questions = pd.read_csv(question_file)
    explains1 = pd.read_csv(explain_file1)
    explains2 = pd.read_csv(explain_file2)

    # 将diff_index转换为set以提高查找效率
    diff_index_set = set(diff_index)

    # 筛选出答案不同的问题
    result = []
    for index in diff_index_set:
        # 获取对应的行
        question_row = questions[questions['id'] == index].iloc[0]
        explain1_row = explains1[explains1['id'] == index].iloc[0]
        explain2_row = explains2[explains2['id'] == index].iloc[0]

        # 提取需要的信息
        question_id = question_row['id']
        question_text = question_row['question']
        answer1 = explain1_row['answer']
        explain1 = explain1_row['explain']
        answer2 = explain2_row['answer']
        explain2 = explain2_row['explain']

        # 将结果添加到列表中
        result.append((question_id, question_text, answer1, explain1, answer2, explain2))

    return result




if __name__ == '__main__':
    # question_file = r'test_A.csv'
    result_file1 = r'result.csv'
    result_file2 = r'result2.csv'
    explain_file1 = r'result_exp.csv'
    explain_file2 = r'result_exp2.csv'
    question_file = r'test_A.csv'

    diff_id = find_different_answers(result_file2, result_file1)

    diff_content = get_judge_statement(question_file, explain_file1, explain_file2, diff_id)

    # 根据diff_id 在explain_file1 和 explain_file2 中得到他们给出理由的原因，
    # 生成
    # i = 0
    for i in range(len(diff_content)):
        print("{},question:{} \n answer1:{} \n explain1: {} \n answer2:{} \n explain2: {}".format(diff_content[i][0],
                                                                                            diff_content[i][1],diff_content[i][2] ,diff_content[i][3],diff_content[i][4],diff_content[i][5] ))
        print('-'*50)
    # print('dd')

# 11 49