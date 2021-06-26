from flask import Flask, request, render_template
import pickle
import pandas as pd
import numpy as np
from sklearn.externals import joblib

app = Flask(__name__,template_folder='template')

model = joblib.load("Students_Marks_Predictor_model.pkl")

df = pd.DataFrame()

@app.route('/',methods=['GET', 'POST'])
def home():
    return render_template("index.html")


@app.route('/predict',methods = ['POST'])
def predict():
    global df

    input_features = [int(x) for x in request.form.values()]
    features_value = np.array(input_features)

    # validate input hours
    if input_features[0] < 0 or input_features[0] > 24:
        return render_template('index.html', Prediction_text = 'Please enter valid hours between 1 to 24')


    output = model.predict([features_value])[0][0].round(2)

    # input and predicted value store in df then save in csv file
    df = pd.concat([df,pd.DataFrame({'Study Hours':input_features, 'Predicted out':[output]})],ignore_index=True)
    print(df)
    df.to_csv('smp_data_from_app.csv')

    return render_template('index.html',Prediction_text = f"you will get {output}% marks, when you do study {input_features} hours per day")


if __name__ == "__main__":
    app.run(debug=True)