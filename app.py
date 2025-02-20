from flask import Flask, request, render_template, jsonify
import torch
import os
import pickle as pkl
from importlib import import_module

# 设置默认参数
UNK, PAD = '<UNK>', '<PAD>'
dataset_name = "THUCNews"  # 设置数据集名称
key = {
    0: 'finance',
    1: 'realty',
    2: 'stocks',
    3: 'education',
    4: 'science',
    5: 'society',
    6: 'politics',
    7: 'sports',
    8: 'game',
    9: 'entertainment'
}

# 预定义两个模型名称
MODEL_NAMES = ["TextCNN", "TextRNN"]  

# 模型和配置字典
models = {}
configs = {}

# 加载模型函数
def init_model(model_name):
    if model_name in models:
        return models[model_name], configs[model_name]
    
    x = import_module('models.' + model_name)
    config = x.Config(dataset_name, embedding='random')
    
    if os.path.exists(config.vocab_path):
        vocab = pkl.load(open(config.vocab_path, 'rb'))
        config.n_vocab = len(vocab)
    
    model = x.Model(config).to(config.device)
    model.load_state_dict(torch.load(config.save_path, map_location=torch.device('cuda')))
    model.eval()

    # 缓存模型和配置
    models[model_name] = model
    configs[model_name] = config
    
    return model, config

# 初始化 Flask 应用
app = Flask(__name__)

# 在启动时加载 TextCNN 和 TextRNN 模型
for model_name in MODEL_NAMES:
    init_model(model_name)

def build_predict_text(text, use_word, config, vocab):
    if use_word:
        tokenizer = lambda x: x.split(' ')
    else:
        tokenizer = lambda x: [y for y in x]

    token = tokenizer(text)
    seq_len = len(token)
    pad_size = config.pad_size
    if pad_size:
        if len(token) < pad_size:
            token.extend([PAD] * (pad_size - len(token)))
        else:
            token = token[:pad_size]
            seq_len = pad_size

    words_line = []
    for word in token:
        words_line.append(vocab.get(word, vocab.get(UNK)))

    ids = torch.LongTensor([words_line]).to(config.device)
    seq_len = torch.LongTensor(seq_len).to(config.device)

    return ids, seq_len

def predict(text, model, config, vocab):
    data = build_predict_text(text, use_word=False, config=config, vocab=vocab)  # 使用字符级别的分词
    with torch.no_grad():
        outputs = model(data)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_index = torch.argmax(probabilities)
        predicted_label = key[int(predicted_index)]
        predicted_probability = probabilities[0, predicted_index].item() * 100

    all_probabilities = {key[i]: probabilities[0, i].item() * 100 for i in range(len(key))}
    return predicted_label, predicted_probability, all_probabilities

# 首页路由，显示文本输入表单
@app.route('/')
def home():
    return render_template('index.html')

# 预测路由，处理表单提交的文本
@app.route('/predict', methods=['POST'])
def make_prediction():
    # 获取用户选择的模型
    selected_model = request.form.get('model', 'TextCNN')
    
    # 获取相应的模型和配置
    model, config = init_model(selected_model)

    # 获取模型的词汇表
    vocab = pkl.load(open(config.vocab_path, 'rb'))

    text = request.form['text']
    if text:
        label, probability, all_probs = predict(text, model, config, vocab)
        return render_template('index.html', label=label, probability=probability, all_probs=all_probs, input_text=text, selected_model=selected_model)
    else:
        return render_template('index.html', error="请输入文本进行预测。", input_text="", selected_model=selected_model)

# 启动 Flask 应用
if __name__ == "__main__":
    app.run()
