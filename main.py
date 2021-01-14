# ----------------------------- 请加载您最满意的模型 -------------------------------
# 加载模型(请加载你认为的最佳模型)
# 加载模型,加载请注意 model_path 是相对路径, 与当前文件同级。
# 如果你的模型是在 results 文件夹下的 temp.pth 模型，则 model_path = 'results/temp.pth'

# 创建模型实例

from logging import debug
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig
import torch


model_path = 'mo5-pl-roberta-ad/best_tfmr'
model = BertForSequenceClassification.from_pretrained(model_path)
model.eval()
tokenizer = BertTokenizer.from_pretrained(model_path)


# model.save_pretrained('./dataset/')
# -------------------------请勿修改 predict 函数的输入和输出-------------------------
def predict(text):
    """
    :param text: 中文字符串
    :return: 字符串格式的作者名缩写
    """
     # ----------- 实现预测部分的代码，以下样例可代码自行删除，实现自己的处理方式 -----------
    # 自行实现构建词汇表、词向量等操作
    # 将句子做分词，然后使用词典将词语映射到他的编号
    # text2idx = [TEXT.vocab.stoi[i] for i in jb.lcut(text) ]
    # 转化为Torch接收的Tensor类型
    # text2idx = torch.Tensor(text2idx).long()

    text = tokenizer(text)['input_ids']
    text = torch.tensor(text)
    results = model(text.unsqueeze(0))[0].view(-1,1).squeeze(0)
    probe = torch.softmax(results, dim=0).detach().numpy().tolist()
    # 模型预测部分
    prediction = torch.argmax(results,0).numpy()
    import IPython; IPython.embed(); exit(1)
    # --------------------------------------------------------------------------

    return prediction.item(), probe


sen = "我听到一声尖叫，感觉到蹄爪戳在了一个富有弹性的东西上。定睛一看，不由怒火中烧。原来，趁着我不在，隔壁那个野杂种——沂蒙山猪刁小三，正舒坦地趴在我的绣榻上睡觉。我的身体顿时痒了起来，我的目光顿时凶了起来。"
sen = "手机别贴脸手机别贴脸手机别贴脸"
predict(sen)