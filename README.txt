lstm：
运行 python lstm_train.py 以获得模型（文件夹里本身已经有训练好的模型"my_model.h5"，所以可以不运行这条命令）
运行 python lstm_predict.py 以获得预测结果（90%训练集和10%测试集）

lstm + word2vec：
运行 python sentence.py 获得句子分词后的文本"sentence.txt"
运行 python word2vec.py 获得word2vec模型"sentence.model"
运行lstm_word2vec.py 获得训练模型与预测结果

lstm + 规则：
运行 python model.py 获得json文件"assignment_test_data_word_segment.py"