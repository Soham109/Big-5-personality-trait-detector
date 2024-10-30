from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)

regressor = joblib.load('personality_predictor.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

@app.route('/')
def home():
    return render_template('chatbox.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json() 
    input_data = data['text']
    
    transformed_input = vectorizer.transform([input_data])
    
    prediction = regressor.predict(transformed_input)
    
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)