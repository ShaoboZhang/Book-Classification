### 1 任务

通过对书籍简介或摘要的理解，训练出一模型，可预测书籍所属类别，如文学、教育等。

### 2 数据

数据源自JD图书中部分书籍的简介，包括文学、社会科学、小说等共33个不同类别。其中训练数据206316条，验证数据58948条，测试数据29474条。

### 3 目录

+ data：该目录用于存放数据，受限于数据大小，若需获取数据，可与我联系
+ lib：该目录用于存放bert-base预训练模型，下载于hugging face
+ model: 该目录存放工程用到的工具函数，包括：
    + config.py：设置工程中的各项参数
    + data_utils.py：数据预处理
    + model.py：设置模型
    + utils.py：模型训练及结果预测
+ main.py：工程主程序


### 4 使用说明

#### 4.1 数据处理
本次工程属于分类任务，因此针对此次数据，使用的数据处理方式包括：中文分词、去除标点、去除停用词。
在config.py中设置```is_preprocess=True/False```可控制模型是否进行数据预处理。

#### 4.2 模型选择
本次工程使用两种不同模型，分别对数据进行训练及比较模型效果。一种为传统Embedding+LSTM模型，一种为Bert+LSTM模型。
在config.py中设置```use_bert=True/False```可选择模型是否使用Bert进行训练。


### 5 实验结果

| 模型选择 | 是否预处理 | 准确率 |
| :-----: | :----: | :----: |
| Embedding+LSTM | 是 <br> 否  | 76.0% <br> 75.6% |
| Bert-base+LSTM | 是 <br> 否  | 82.2% <br> 80.4% |
