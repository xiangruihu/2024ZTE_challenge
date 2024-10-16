# import dotenv
# dotenv.load_dotenv()

import requests
# from langchain.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter

loader = PyPDFLoader(r'D:\python\ZTE_challenge\data\cn202306.pdf')

documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)

print('d')
# from langchain.embeddings import OpenAIEmbeddings
# from langchain.vectorstores import Weaviate
# import weaviate
# from weaviate.embedded import EmbeddedOptions
#
# client = weaviate.Client(
#   embedded_options = EmbeddedOptions()
# )
#
# vectorstore = Weaviate.from_documents(
#     client = client,
#     documents = chunks,
#     embedding = OpenAIEmbeddings(),
#     by_text = False
# )


from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import dashscope
dashscope.api_key = ''

p = pipeline(
    task=Tasks.document_segmentation,
    model='damo/nlp_bert_document-segmentation_chinese-base')

result = p(documents=r'移动端语音唤醒模型，检测关键词为“小云小云”。模型主体为4层FSMN结构，使用CTC训练准则，参数量750K，适用于移动端设备运行。模型输入为Fbank特征，输出为基于char建模的中文全集token预测，测试工具根据每一帧的预测数据进行后处理得到输入音频的实时检测结果。模型训练采用“basetrain + finetune”的模式，basetrain过程使用大量内部移动端数据，在此基础上，使用1万条设备端录制安静场景“小云小云”数据进行微调，得到最终面向业务的模型。后续用户可在basetrain模型基础上，使用其他关键词数据进行微调，得到新的语音唤醒模型，但暂时未开放模型finetune功能。')

print(result[OutputKeys.TEXT])