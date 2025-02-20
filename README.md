# 文本分类系统（深度学习模型）

该项目是基于深度学习的文本分类系统，旨在对中文文本进行多类别分类。系统使用 **TextCNN** 和 **TextRNN** 模型，并提供一个Web界面，允许用户实时与系统进行交互。

## 使用的技术

- **后端：** Flask, PyTorch
- **前端：** HTML, CSS
- **深度学习模型：** TextCNN, TextRNN
- **数据处理：** NumPy, Pickle
- **部署：** Web API（Flask）

## 功能特点

1. **数据预处理：**
   - 支持单词级别和字符级别的分词。
   - 根据训练数据集构建词汇表，支持处理未知词（`<UNK>`）和填充（`<PAD>`）。
   - 对文本进行填充或截断，确保输入文本长度一致。

2. **模型训练与评估：**
   - 实现 **TextCNN** 和 **TextRNN** 两种深度学习模型进行文本分类。
   - 使用交叉熵损失函数和Adam优化器进行训练。
   - 实现早停机制以防止过拟合，并根据验证集损失保存最优模型。

3. **Web接口：**
   - 基于Flask的Web应用程序，允许用户输入文本并选择分类模型（TextCNN或TextRNN）。
   - 返回实时预测结果，包括预测标签及其概率，以及所有类别的概率分布。

4. **用户体验：**
   - 简洁直观的前端设计，具有平滑的用户交互效果。
   - 背景图和动态表单效果，提升用户体验。

## 安装步骤

1. 克隆该项目：

    ```bash
    git clone https://github.com/ycsun0v0/TextClassificationSystem.git
    ```

2. 安装所需的依赖：

    ```bash
    pip install -r requirements.txt
    ```

3. 启动Flask应用程序：

    ```bash
    python app.py
    ```

4. 打开浏览器访问：

    ```
    http://127.0.0.1:5000
    ```

## 模型

- **TextCNN：** 卷积神经网络模型，使用多个不同尺寸的卷积滤波器从文本序列中提取特征。
- **TextRNN：** 循环神经网络模型，使用双向LSTM捕捉文本序列中的长期依赖关系。


# Text Classification System (Deep Learning Models)

This project is a deep learning-based text classification system developed for classifying Chinese texts into multiple categories. The system employs models such as **TextCNN** and **TextRNN** and provides a web interface for users to interact with the system in real-time.

## Technologies Used

- **Backend:** Flask, PyTorch
- **Frontend:** HTML, CSS
- **Deep Learning Models:** TextCNN, TextRNN
- **Data Handling:** NumPy, Pickle
- **Deployment:** Web API (Flask)

## Features

1. **Data Preprocessing:**
   - Supports word-level and character-level tokenization.
   - Builds a vocabulary from the training dataset, with support for handling unknown words (`<UNK>`) and padding (`<PAD>`).
   - Texts are padded or truncated to a fixed length to standardize input size.

2. **Model Training and Evaluation:**
   - Implements **TextCNN** and **TextRNN** models for text classification.
   - Uses Cross-Entropy loss and Adam optimizer for training.
   - Implements early stopping to prevent overfitting and saves the best model based on validation loss.

3. **Web Interface:**
   - Flask-based web application allowing users to input text and choose a classification model (TextCNN or TextRNN).
   - Returns real-time prediction with the predicted label and its probability, along with all categories’ probability distribution.

4. **User Experience:**
   - Clean and intuitive front-end design with smooth UI interactions.
   - Background image and dynamic form field effects for better user engagement.

## Installation

1. Clone this repository:

    ```bash
    git clone https://github.com/ycsun0v0/TextClassificationSystem.git
    ```

2. Install required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Start the Flask application:

    ```bash
    python app.py
    ```

4. Open your browser and visit:

    ```
    http://127.0.0.1:5000
    ```

## Models

- **TextCNN:** Convolutional neural network model that uses multiple convolutional filters of different sizes to extract features from text sequences.
- **TextRNN:** Recurrent neural network model using bidirectional LSTM to capture long-term dependencies in text sequences.


