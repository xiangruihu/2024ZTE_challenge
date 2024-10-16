import csv
import re
import pandas as pd
from qwen_model.qwen import get_completion, build_judgement_extract
# load_questions

# 将问题放置到一个列表当中

def load_questions(file_path):
    questions_dict = {}
    with open(file_path, mode='r', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # 跳过标题行，如果没有标题行可以注释掉这一行
        for row in csv_reader:
            id = int(row[0])
            question = row[1]
            questions_dict[id] = question
    return questions_dict




def save_result(csv_file, id, answer):
    # 检查文件是否存在
    file_exists = False
    try:
        with open(csv_file, mode='r', encoding='utf-8') as file:
            file_exists = True
    except FileNotFoundError:
        pass

    # 打开文件进行写入操作
    with open(csv_file, mode='a', encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            # 如果文件不存在，写入标题行
            writer.writerow(['id', 'answer'])
        # 写入新的一行数据
        writer.writerow([id, answer])

# save_result('result.csv', '0', 'T')



def save_result_explain(csv_file, id, answer, explain):
    # 检查文件是否存在
    file_exists = False
    try:
        with open(csv_file, mode='r', encoding='utf-8') as file:
            file_exists = True
    except FileNotFoundError:
        pass

    # 打开文件进行写入操作
    with open(csv_file, mode='a', encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            # 如果文件不存在，写入标题行
            writer.writerow(['id', 'answer', 'explain'])
        # 写入新的一行数据
        writer.writerow([id, answer, explain])


def extract_answer_and_reason(response_text, model_name = 'qwen-14b-chat'):
    # 使用正则表达式匹配答案（T或F）


    # 需要对
    # response_copy = response_text
    # extract_template = build_judgement_extract()
    # extract_message = extract_template.format_messages(reson = response_text)
    #
    # response_text = get_completion(extract_message[0].content, model_name)

    answer_match = re.search(r'\b(T|F|N)\b', response_text)
    answer = answer_match.group(0) if answer_match else None

    # # 提取理由，找到第一个换行符后的一段文本
    # reason_start_index = response_text.find('\n') + 1
    # reason_end_index = response_text.find('\n', reason_start_index)
    # reason = response_text[reason_start_index:reason_end_index].strip() if reason_end_index != -1 else response_text[
    #                                                                                                     reason_start_index:].strip()

    return answer, response_text.replace('\n','')


def extract_keywords(text):
    matches = re.findall(r'"(.*?)"', text)
    return matches

def find_different_answers(file1, file2):
    """
    读取两个CSV文件，找到answer列不同的题目ID。

    参数:
    file1 (str): 第一个CSV文件的路径
    file2 (str): 第二个CSV文件的路径

    返回:
    list: answer列不同的题目ID
    """
    # 读取CSV文件
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    # 合并两个数据框，按id列合并
    merged_df = pd.merge(df1, df2, on='id', suffixes=('_file1', '_file2'))

    # 找到answer列不同的行
    different_answers = merged_df[merged_df['answer_file1'] != merged_df['answer_file2']]

    # 取出不同的id
    different_ids = different_answers['id'].tolist()

    return different_ids

