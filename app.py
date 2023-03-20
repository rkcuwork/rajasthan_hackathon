import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [x for x in request.form.values()]
    # final_features = [np.array(int_features)]
    # print(final_features)
    text1 = int_features[0]
    text2 = int_features[1]
    matrix = model.fit_transform([text1,text2])

    prediction = cosine_similarity(matrix[0:1],matrix)[0][1]

    output = round(prediction, 2)

    return render_template('index.html', prediction_text='Employee Salary should be $ {}'.format(output))
    # return render_template('index.html', prediction_text='Employee Salary should be $ {}'.format(len(final_features)))


if __name__ == "__main__":
    app.run(debug=True)