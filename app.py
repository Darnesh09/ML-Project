from flask import Flask,redirect,url_for,render_template,request
import pandas as pd
import numpy as np
from src.piplines.predict_pipline import CustomData,PredictPipeline

app = Flask(__name__)

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data = CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('reading_score')),
            writing_score=float(request.form.get('writing_score'))
        )

        df = data.get_data_as_dataframe()
        pred_obj = PredictPipeline()
        result = pred_obj.predict(df)

        return render_template('home.html',results=min(result[0],100),form_data=request.form)


if __name__=='__main__':
    app.run(debug=True,host='0.0.0.0')