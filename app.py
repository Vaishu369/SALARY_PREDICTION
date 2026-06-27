import numpy as np
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS  

app = Flask(__name__)
CORS(app)  
model = joblib.load(r"C:\Users\rahitya\OneDrive\Desktop\salary prediction\salary_model.pkl")
job_role_mapping = {
    "Data Scientist": [1, 0, 0, 0],  
    "Software Engineer": [0, 1, 0, 0],
    "Product Manager": [0, 0, 1, 0],
    "Analyst": [0, 0, 0, 1]
}

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        experience = np.array([[data['experience']]])
        job_role = data['jobRole']

        if job_role not in job_role_mapping:
            return jsonify({"error": "Invalid job role!"})
        job_role_encoded = np.array([job_role_mapping[job_role]])

        input_data = np.hstack((experience, job_role_encoded))

        predicted_salary = model.predict(input_data)

        return jsonify({"predicted_salary": round(predicted_salary[0], 2)})
    
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
