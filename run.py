# coding: UTF-8
import time
import torch
import numpy as np
from train_eval import train, init_network  # 引入训练和网络初始化函数
from importlib import import_module  # 动态导入模块
import argparse  # 用于命令行参数解析

# 设置命令行参数解析器
parser = argparse.ArgumentParser(description='Chinese Text Classification')  # 设置命令行工具的描述信息
# 添加--model参数，指定选择的模型类型
parser.add_argument('--model', type=str, required=True, help='choose a model: TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer')
# 添加--embedding参数，选择词嵌入类型：random（随机初始化）或pre_trained（使用预训练嵌入）
parser.add_argument('--embedding', default='pre_trained', type=str, help='random or pre_trained')
# 添加--word参数，选择是使用词级别（True）还是字符级别（False）
parser.add_argument('--word', default=False, type=bool, help='True for word, False for char')
# 解析命令行参数
args = parser.parse_args()

if __name__ == '__main__':
    dataset = 'THUCNews'  # 设置使用的数据集，这里是THUCNews

    # 根据不同的embedding选择，设置不同的嵌入文件
    embedding = 'embedding_SougouNews.npz'  # 默认使用搜狗新闻嵌入,65000条新闻数据
    if args.embedding == 'random':
        embedding = 'random'  # 如果选择随机初始化嵌入，则设置为'random'
    
    model_name = args.model  # 获取选择的模型名称（例如TextCNN, TextRNN等）
    
    # 如果使用FastText模型，需要导入相关工具和模块
    if model_name == 'FastText':
        from utils_fasttext import build_dataset, build_iterator, get_time_dif  # FastText相关的工具
        embedding = 'random'  # FastText模型不使用预训练嵌入，直接随机初始化
    else:
        from utils import build_dataset, build_iterator, get_time_dif  # 其他模型的通用工具

    # 动态导入指定的模型（通过import_module按需导入）
    x = import_module('models.' + model_name)
    config = x.Config(dataset, embedding)  # 使用模型对应的配置，传入数据集和嵌入文件
    np.random.seed(1)  # 设置NumPy的随机种子，确保结果可复现
    torch.manual_seed(1)  # 设置PyTorch的CPU随机种子
    torch.cuda.manual_seed_all(1)  # 设置PyTorch的GPU随机种子（适用于多GPU环境）
    torch.backends.cudnn.deterministic = True  # 设置CUDNN为确定性模式，确保每次结果一致

    # 记录程序开始时间，用于计算运行时长
    start_time = time.time()
    print("Loading data...")  # 打印提示信息，表示数据加载开始
    
    # 使用build_dataset函数加载数据集，返回词汇表和训练、验证、测试数据
    vocab, train_data, dev_data, test_data = build_dataset(config, args.word)
    
    # 构建数据迭代器，供训练使用
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    
    # 计算数据加载所花费的时间
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)  # 打印数据加载的时间

    # 开始训练部分
    config.n_vocab = len(vocab)  # 设置词汇表大小
    model = x.Model(config).to(config.device)  # 实例化模型并将其移动到指定设备（GPU或CPU）

    # 如果选择的模型不是Transformer类型，则初始化网络参数
    if model_name != 'Transformer':
        init_network(model)

    print(model.parameters)  # 打印模型的参数信息
    train(config, model, train_iter, dev_iter, test_iter)  # 调用train函数开始训练模型
