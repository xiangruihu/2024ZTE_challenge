from dataloader.pdf_loader import get_chunks, load_cn_data
# from langchain.text_splitter import

# 想要做的是清洗chunks

# txts = '这里是电子科技大学团队，这里有很多聪明的人，大家都很努力，为未来拼搏，希望能够取得好的成绩和结果'

#
pdf_path = r'/home/disks/sdd/hxr/hxr/python/ZTE_challenge/data/cn202303.pdf'
cn_texts = load_cn_data(pdf_path)

# chunks = get_chunks(cn_texts, chunk_size=160, overlap_size= 40)




def chunks_cleaner(chunks,model, chunk_size = 5, chunk_windows = 3):

    # 写一个基于滑动窗口的多个chunks 融合机制

    return 0

def chunks_cleaner_windows_based(chunks, model):

    return 0