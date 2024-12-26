import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# 1. 加载数据集
data_path = os.path.join(os.getcwd(), 'xunlian.csv')
data = pd.read_csv(data_path)
# 2. 检查缺失值
# 通过isnull()方法检查数据集中每个元素是否为缺失值（NaN），然后使用sum()方法对每列的缺失值数量进行求和统计
# 最后打印输出各列的缺失值统计情况，以便了解数据的完整性，确定是否需要进行相应的数据处理
print(f"缺失值统计:\n{data.isnull().sum()}")

# 3. 数据预处理
# 去掉缺失值
# 调用dropna()方法删除包含缺失值的行，subset参数指定了仅在'text'（邮件内容）列和'label'（邮件标签）列有缺失值的行才会被删除
# inplace=True表示直接在原数据data上进行修改，而不是返回一个新的数据副本
data.dropna(subset=['text', 'label'], inplace=True)

# 提取邮件内容和标签
# 将数据集中名为'text'的列作为邮件内容数据，赋值给变量X，这将是后续进行文本特征提取和模型训练的输入数据之一
X = data['text']
# 将数据集中名为'label'的列作为邮件标签数据，赋值给变量y，其中通常以1表示垃圾邮件，0表示非垃圾邮件，作为模型训练的目标输出
y = data['label']

# 4. 数据集划分为训练集和测试集
# 使用sklearn库的train_test_split函数将数据集按照指定比例划分为训练集和测试集
# X表示输入特征数据（邮件内容），y表示对应的目标标签（是否为垃圾邮件）
# test_size=0.3表示测试集占总数据集的比例为30%，即划分出30%的数据作为测试集，剩下70%作为训练集
# random_state=42是随机种子，设置它可以保证每次划分数据集的结果都一致，便于复现实验结果
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 5. 文本特征提取（TF-IDF）
# 创建TfidfVectorizer对象，用于将文本数据转换为TF-IDF（词频-逆文档频率）特征向量表示
# stop_words='english'表示去除英文中的常见停用词，这些词通常对文本分类的帮助不大，去除可以减少特征维度并提高模型性能
# max_features=5000指定了最多保留5000个最具代表性的特征（单词或词组），通过这种方式可以控制特征向量的维度，避免维度灾难并提高计算效率
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
# 使用训练集数据进行拟合（学习词汇表以及计算每个单词的TF-IDF权重等），并将训练集文本数据转换为TF-IDF特征向量矩阵
X_train_tfidf = vectorizer.fit_transform(X_train)
# 依据在训练集上拟合好的词汇表等信息，将测试集文本数据转换为TF-IDF特征向量矩阵，使其与训练集的特征表示形式一致，便于后续模型进行预测
X_test_tfidf = vectorizer.transform(X_test)

# 6. 模型选择与训练
# 使用朴素贝叶斯分类器中的多项式朴素贝叶斯（MultinomialNB），它适用于处理文本数据这类离散特征的分类问题，比如基于词频等特征进行文本分类
nb_classifier = MultinomialNB()
# 使用训练集的TF-IDF特征向量矩阵和对应的标签数据对朴素贝叶斯分类器进行训练，让模型学习邮件文本特征与是否为垃圾邮件之间的关系
nb_classifier.fit(X_train_tfidf, y_train)

# 7. 预测与评估
# 使用训练好的朴素贝叶斯分类器对测试集的TF-IDF特征向量矩阵进行预测，得到预测的标签结果，存储在y_pred变量中
y_pred = nb_classifier.predict(X_test_tfidf)
# 计算预测结果的准确率，通过比较预测标签y_pred和真实标签y_test，使用accuracy_score函数计算预测正确的样本占总样本的比例
accuracy = accuracy_score(y_test, y_pred)
# 打印输出朴素贝叶斯分类器在测试集上的准确率，格式化输出为保留两位小数的百分数形式
print(f'朴素贝叶斯分类器的准确率: {accuracy * 100:.2f}%')

# 混淆矩阵与分类报告
# 打印混淆矩阵，混淆矩阵展示了分类模型在预测过程中的具体情况，包括真正例（True Positive）、假正例（False Positive）、真反例（True Negative）、假反例（False Negative）的数量
# 可以直观地看出模型在不同类别上的预测准确性和误分类情况
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
# 打印分类报告，分类报告包含了更详细的分类评估指标，如准确率（accuracy）、精确率（precision）、召回率（recall）和F1分数（F1-score）等信息
# 对于每个类别（这里是垃圾邮件和非垃圾邮件两类）都有相应的指标值，能全面地评估模型的分类性能
print("Classification Report:")
print(classification_report(y_test, y_pred))

# 8. 其他模型：支持向量机（SVM）与随机森林
# 创建支持向量机（SVC）分类器对象，这里选择线性核函数（kernel='linear'），不同的核函数适用于不同的数据分布和分类场景，线性核函数相对简单且在很多线性可分或近似线性可分的数据上表现良好
svm_classifier = SVC(kernel='linear')
# 使用训练集的TF-IDF特征向量矩阵和对应的标签数据对支持向量机分类器进行训练
svm_classifier.fit(X_train_tfidf, y_train)
# 使用训练好的支持向量机分类器对测试集的TF-IDF特征向量矩阵进行预测，得到预测结果
svm_pred = svm_classifier.predict(X_test_tfidf)
# 打印输出支持向量机模型在测试集上的准确率，格式与前面朴素贝叶斯准确率输出类似
print("\n支持向量机模型的准确率:")
print(f'Accuracy: {accuracy_score(y_test, svm_pred) * 100:.2f}%')

# 创建随机森林分类器对象，n_estimators=100表示构建包含100棵决策树的随机森林，随机森林通过集成多个决策树的预测结果来提高分类性能和稳定性
# random_state=42用于设置随机种子，保证每次运行代码构建的随机森林结构在一定程度上可重复
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
# 使用训练集的TF-IDF特征向量矩阵和对应的标签数据对随机森林分类器进行训练
rf_classifier.fit(X_train_tfidf, y_train)
# 使用训练好的随机森林分类器对测试集的TF-IDF特征向量矩阵进行预测，得到预测结果
rf_pred = rf_classifier.predict(X_test_tfidf)
# 打印输出随机森林模型在测试集上的准确率
print("\n随机森林模型的准确率:")
print(f'Accuracy: {accuracy_score(y_test, rf_pred) * 100:.2f}%')

# 9. 模型调优：网格搜索优化朴素贝叶斯
# 定义一个参数字典param_grid，用于指定要搜索的超参数及其取值范围，这里是对多项式朴素贝叶斯的'alpha'参数进行搜索
# 'alpha'是平滑参数，控制模型的复杂度，不同的取值可能影响模型的泛化能力和在测试集上的性能表现
param_grid = {'alpha': [0.5, 1.0, 1.5, 2.0]}
# 创建GridSearchCV对象，用于进行网格搜索交叉验证来寻找最优超参数组合
# 第一个参数MultinomialNB()是要优化的模型对象，即多项式朴素贝叶斯分类器
# param_grid是要搜索的超参数范围字典
# cv=5表示采用5折交叉验证，即将训练集分成5份，每次用其中4份作为训练数据，1份作为验证数据，轮流进行5次，综合评估模型性能
# n_jobs=-1表示使用所有可用的CPU核心进行并行计算，加快搜索速度（如果电脑是多核处理器的话）
grid_search = GridSearchCV(MultinomialNB(), param_grid, cv=5, n_jobs=-1)
# 使用训练集的TF-IDF特征向量矩阵和对应的标签数据进行网格搜索，找到最优的超参数组合以及对应的最优模型
grid_search.fit(X_train_tfidf, y_train)
# 打印输出通过网格搜索找到的最佳超参数组合
print(f"\n最佳超参数: {grid_search.best_params_}")
# 获取最佳模型（即使用最佳超参数组合的多项式朴素贝叶斯分类器）
best_nb_classifier = grid_search.best_estimator_
# 使用最佳模型对测试集的TF-IDF特征向量矩阵进行预测，得到预测结果
best_nb_pred = best_nb_classifier.predict(X_test_tfidf)
# 计算并打印输出使用优化后的朴素贝叶斯模型在测试集上的准确率
print(f'优化后的朴素贝叶斯准确率: {accuracy_score(y_test, best_nb_pred) * 100:.2f}%')

# 10. 数据可视化 - 词云图
# 从数据集中筛选出标签为1（即垃圾邮件）的所有邮件内容，将它们合并成一个长字符串，每个邮件内容之间用空格隔开
# 这样后续可以基于这个字符串生成垃圾邮件相关的词云，展示垃圾邮件中常见的词汇情况
spam_words = ' '.join(data[data['label'] == 1]['text'])
# 同样地，从数据集中筛选出标签为0（即非垃圾邮件）的所有邮件内容，合并成一个长字符串，用于生成非垃圾邮件的词云
ham_words = ' '.join(data[data['label'] == 0]['text'])

# 创建垃圾邮件和非垃圾邮件的词云
# 使用WordCloud类创建一个词云对象，设置词云的宽度为800像素，高度为400像素，背景颜色为黑色
# 然后调用generate()方法，传入垃圾邮件的文本字符串，生成能够反映垃圾邮件中词汇出现频率的词云图（通过字体大小等体现频率高低）
spam_wordcloud = WordCloud(width=800, height=400, background_color='black').generate(spam_words)
# 类似地，创建非垃圾邮件的词云对象，背景颜色设置为白色，传入非垃圾邮件的文本字符串生成相应词云图
ham_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(ham_words)

# 显示词云图
# 创建一个新的图形对象，设置图形的大小为宽12英寸、高8英寸
plt.figure(figsize=(12, 8))
# 在图形中添加第一个子图，将其划分为1行2列的布局中的第1个位置
plt.subplot(1, 2, 1)
# 在该子图中显示垃圾邮件的词云图，interpolation='bilinear'用于设置图像插值方式，使显示效果更平滑
plt.imshow(spam_wordcloud, interpolation='bilinear')
# 设置子图的标题为"垃圾邮件词云"
plt.title("垃圾邮件词云")
# 关闭坐标轴显示，因为词云图不需要坐标轴来展示信息
plt.axis('off')

# 在图形中添加第二个子图，位于1行2列布局中的第2个位置
plt.subplot(1, 2, 2)
# 显示非垃圾邮件的词云图
plt.imshow(ham_wordcloud, interpolation='bilinear')
# 设置子图标题为"非垃圾邮件词云"
plt.title("非垃圾邮件词云")
# 同样关闭坐标轴显示
plt.axis('off')

# 显示整个包含两个词云图的图形
plt.show()

# 11. 保存与加载模型
# 保存模型
# 使用joblib库的dump函数将优化后的最佳朴素贝叶斯分类器模型保存到本地文件'spam_classifier_model.pkl'中
# 这样后续可以直接加载该模型进行垃圾邮件预测，无需重新训练，节省时间并方便部署应用
joblib.dump(best_nb_classifier, 'spam_classifier_model.pkl')
# 同时，将用于文本特征提取的TfidfVectorizer对象也保存到本地文件'vectorizer.pkl'中，因为在进行预测时需要使用相同的特征提取方式将新的邮件文本转换为特征向量
joblib.dump(vectorizer, 'vectorizer.pkl')

# 加载模型
# 使用joblib库的load函数从本地文件'spam_classifier_model.pkl'中加载之前保存的最佳朴素贝叶斯分类器模型
loaded_classifier = joblib.load('spam_classifier_model.pkl')
# 从本地文件'vectorizer.pkl'中加载保存的TfidfVectorizer对象，用于后续对新邮件文本进行特征提取
loaded_vectorizer = joblib.load('vectorizer.pkl')

# 测试加载的模型
# 定义一个函数用于测试加载后的模型对给定邮件文本的预测效果
def predict_spam(email_text):
    # 使用加载的TfidfVectorizer对象将输入的邮件文本转换为TF-IDF特征向量，注意这里需要将邮件文本放在列表中传入，因为transform方法接收的是可迭代对象
    email_tfidf = loaded_vectorizer.transform([email_text])
    # 使用加载的分类器对转换后的特征向量进行预测，得到预测结果（1表示垃圾邮件，0表示非垃圾邮件）
    prediction = loaded_classifier.predict(email_tfidf)
    if prediction == 1:
        print("这是垃圾邮件")
    else:
        print("这不是垃圾邮件")

# 测试加载的模型
# 定义一个示例邮件文本，用于测试加载后的模型是否能正确进行预测
sample_email = "Congratulations! You've won a $1000 gift card. Claim your prize now!"
# 调用predict_spam函数对示例邮件文本进行预测并输出相应的判断结果
predict_spam(sample_email)