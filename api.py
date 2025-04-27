
from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load(r"D:\last year\DSS\Model Files\xgb_model.pkl")
scaler = joblib.load(r"D:\last year\DSS\Model Files\scaler.pkl")

# راوت رئيسي للتجربة
@app.route('/')
def home():
    return 'API is working! '

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # قراءة البيانات المرسلة
        data = request.get_json()

        # استخراج الفيتشرز المطلوبة
        features = np.array(data['features']).reshape(1, -1)

        # عمل scaling
        features_scaled = scaler.transform(features)

        # التنبؤ
        prediction = model.predict(features_scaled)

        # إرسال النتيجة
        return jsonify({'prediction': int(prediction[0])})

    except Exception as e:
        return jsonify({'error': str(e)})

# تشغيل السيرفر
if __name__ == '__main__':
    app.run(debug=True)