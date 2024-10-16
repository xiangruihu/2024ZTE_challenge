import os
import re

# from langchain.document_loaders import PyPDFLoader
import fitz
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredPDFLoader, PDFMinerLoader
# from langchain_community.document_loaders import UnstructuredPDFLoader
# from langchain_community.document_loaders import UnstructuredAPIFileLoader

# from langchain.



def clean_up_text(text):
    # \s 表示任意空白字符， 包括空格 制表符 换行符
    # TODO 无法分段 后续进行更改 段落无法分割出来
    # text = re.sub(r'\s+', ' ', text)
    # 第一步，删除 \n 转义符
    text = re.sub(r'[\r\n]+', '', text)
    # 当有多个空格时删除掉多余的空格
    text = re.sub(r'\s+', ' ',text).strip()
    # 去除特殊字符
    # text = re.sub(r'[^\w\s]', '', text)
    # text = re.sub(r'[^\w\s,.，“‘”’。《》、‘“”！？!?]', '', text)
    text = re.sub(r'[^\w\s,。. ? ! ‘ ’ “ ” , ; :  ？\'  ( ) ...  / \ [ ] { } | * & _ — - ！ ， 、.，“‘”’。《》、‘“”！？!? ； ： ‘ ’ “ ” （） …… —— － · ﹏﹏ ～]', '', text)

    # text =
    # en_txt = r'''[. ? ! , ; : ‘ ’ “ ” ( ) ... — - ' / \ [ ] { } | * & _]'''

    return text


def load_cn_data(pdf_file_path):
    text = ''
    doc = fitz.open(pdf_file_path)
    for page_num in range(len(doc)):  # 按照页码提取doc
        page = doc[page_num]
        text += page.get_text()
    # 对数据进行进一步的清理
    # loader = PDFMinerLoader(pdf_file_path)
    # loader = UnstructuredPDFLoader(pdf_file_path)
    # data_unstru = loader.load()
    # data_unstru_text = data_unstru[0].page_content


    # text = clean_up_text(data_unstru_text)
    text = clean_up_text(text)


    return text
# text_data = load_cn_data(r'D:\python\ZTE_challenge\clean_data\en\ZTE TECHNOLOGIES (No.4) 2023.pdf')
# print('d')
# load_cn_data(r"D:\python\ZTE_challenge\clean_data\cn\cn202303.pdf")



def get_all_document(data_dir):
    pdf_names = os.listdir(data_dir)
    data_base = {}

    for i, pdf_name in enumerate(pdf_names):

        pdf_path = os.path.join(data_dir, pdf_name)
        if os.path.exists(pdf_path):
            print('loading the data of {}'.format(pdf_path))
            txt_data = load_cn_data(pdf_path)
            data_base[pdf_name] = txt_data



    return data_base


def get_chunks(data,chunk_size, overlap_size):
    # TODO 修改chunk 的获取方式
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap_size)
    txts = text_splitter.create_documents([data])

    return txts


def get_all_chunks(data_dict, chunk_size, overlap_size):
    all_chunks = {}

    key_names = list(data_dict.keys())
    for key_name in key_names:
        print(' chuncking {}...'.format(key_name))
        all_chunks[key_name] = get_chunks(data_dict[key_name], chunk_size, overlap_size)

    return all_chunks

# if __name__ == '__main__':
#
#     data_dir = '../clean_data/cn'
#     pdf_names = os.listdir(data_dir)
#     data_base = {}
#
#     for i, pdf_name in enumerate(pdf_names):
#
#         pdf_path = os.path.join(data_dir, pdf_name)
#         if os.path.exists(pdf_path):
#             print('loading the data of {}'.format(pdf_path))
#             txt_data = load_cn_data(pdf_path)
#
#             # 然后需要对文本数据进行解析
#             data_base[pdf_name] = txt_data
#
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
#     txts = text_splitter.create_documents([txt_data])
#     print('d')

