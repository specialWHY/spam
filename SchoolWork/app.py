from flask import Flask, render_template, request
import joblib

# 初始化Flask应用
app = Flask(__name__)

# 加载模型和向量化器
classifier = joblib.load('spam_classifier_model.pkl')  # 加载垃圾邮件分类器模型
vectorizer = joblib.load('vectorizer.pkl')  # 加载用于特征提取的TfidfVectorizer

# 预测垃圾邮件函数
def predict_spam(email_text):
    # 将邮件文本转换为TF-IDF特征向量
    email_tfidf = vectorizer.transform([email_text])
    # 使用模型进行预测
    prediction = classifier.predict(email_tfidf)
    return "垃圾邮件" if prediction == 1 else "非垃圾邮件"

# 路由：显示首页
@app.route('/')
def home():
    return render_template('index.html')

# 路由：处理表单提交
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # 获取用户输入的邮件文本
        email_text = request.form['email_text']
        # 使用预测函数进行分类
        result = predict_spam(email_text)
        return render_template('index.html', result=result)

# 运行Flask应用
if __name__ == '__main__':
    app.run(debug=True)
