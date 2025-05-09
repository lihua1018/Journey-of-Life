from flask import Flask, request, jsonify, render_template_string
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import random

app = Flask(__name__)

# HTML 模板
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>心情地點推薦</title>
    <meta charset="utf-8">
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        .form-group { margin-bottom: 15px; }
        input[type="text"] { width: 100%; padding: 8px; margin-top: 5px; }
        button { padding: 10px 20px; background-color: #4CAF50; color: white; border: none; cursor: pointer; }
        #result { margin-top: 20px; padding: 15px; border: 1px solid #ddd; display: none; }
    </style>
</head>
<body>
    <h1>心情地點推薦系統</h1>
    <div class="form-group">
        <label for="message">請描述你的心情：</label>
        <input type="text" id="message" placeholder="例如：今天感覺很放鬆">
    </div>
    <button onclick="getRecommendation()">取得推薦</button>
    <div id="result"></div>

    <script>
    function getRecommendation() {
        const message = document.getElementById('message').value;
        fetch('/recommend', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({message: message})
        })
        .then(response => response.json())
        .then(data => {
            const resultDiv = document.getElementById('result');
            resultDiv.style.display = 'block';
            resultDiv.innerHTML = `
                <h3>推薦結果</h3>
                <p>地點：${data.location}</p>
                <p>地區：${data.region}</p>
                <p>相關心情：${data.mood}</p>
                <p>相似度：${(data.score * 100).toFixed(2)}%</p>
            `;
        })
        .catch(error => {
            console.error('Error:', error);
            alert('發生錯誤，請稍後再試');
        });
    }
    </script>
</body>
</html>
"""

# 載入模型
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# 載入資料（包含 embedding 欄位）
df = pd.read_csv("data_0509_onehot.csv")
df["embedding"] = df["mood_embedding"].apply(eval)

# 設定隨機種子
random.seed(42)

@app.route("/")
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route("/recommend", methods=["POST"])
def recommend():
    user_input = request.json.get("message", "")
    if not user_input:
        return jsonify({"error": "Missing message"}), 400

    # 使用 Sentence Transformers 取得輸入 embedding
    input_vector = model.encode([user_input])[0].reshape(1, -1)

    # 計算相似度
    db_vectors = np.vstack(df["embedding"].values)
    scores = np.dot(input_vector, db_vectors.T)[0] / (np.linalg.norm(input_vector) * np.linalg.norm(db_vectors, axis=1))
    
    # 取得前5個最相似的結果的索引
    top_n = 5
    top_indices = np.argsort(scores)[-top_n:][::-1]
    
    # 從前5個結果中隨機選擇一個
    selected_index = random.choice(top_indices)
    
    result = {
        "location": df.iloc[selected_index]["新增地點"],
        "region": df.iloc[selected_index]["請選擇臺灣某地區"],
        "mood": df.iloc[selected_index]["心情標籤"],
        "score": float(scores[selected_index])
    }
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)