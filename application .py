import pickle
from flask import Flask, request, jsonify,render_template
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

app = Flask(__name__)
@app.route('/')
def home():
    return "Welcome to the Model Training API!"
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        try:
            data = CustomData(
                gender=request.form.get('gender'),
                race_ethnicity=request.form.get('race_ethnicity'),
                parental_level_of_education=request.form.get('parental_level_of_education'),
                lunch=request.form.get('lunch'),
                test_preparation_course=request.form.get('test_preparation_course'),
                math_score=int(request.form.get('math_score')),
                reading_score=int(request.form.get('reading_score')),
                writing_score=int(request.form.get('writing_score'))
            )
            df = data.get_data_as_dataframe()
            predict_pipeline = PredictPipeline()
            predictions = predict_pipeline.predict(df)
            return render_template('home.html', prediction_text='Predicted Score: {}'.format(predictions[0]))
        except Exception as e:
            import traceback
            traceback.print_exc()
            return render_template('home.html', error_text='Error occurred: {}'.format(str(e)))
# if __name__ == "__main__":
#     app.run(debug=True)