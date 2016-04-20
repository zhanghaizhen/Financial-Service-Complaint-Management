from flask import Flask,request,render_template
app = Flask(__name__)
import cPickle as pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pandas as pd



# home page
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/submit_as_business')
def submit_as_business():
    return '''<form action="/disput_predict" method='POST' >
        <input type="text" name="business_input" />
        <input type="submit" />
    </form>'''

@app.route('/submit_as_consumer')
def submit_as_consumer():
    return '''<form action="/response_predict" method='POST' >
        <input type="text" name="consumer_input" />
        <input type="submit" />
    </form>'''

@app.route('/dispute_predict', methods=['POST'] )
def dispute_predict():
    text = str(request.form['business_input'])

    with open('data/vectorizer.pkl') as f:
        vectorizer = pickle.load(f)

    with open('data/model_dispute.pkl') as f:
        model_dispute = pickle.load(f)

    X = vectorizer.transform([text])
    predictions= model_dispute.predict(X)
    for prediction in predictions:
        if prediction==True:
            return "Likely to Dispute"
        if prediction==False:
            return "Unlikely to Dispute"

@app.route('/response_predict', methods=['POST'] )
def response_predict():
    text = str(request.form['consumer_input'])

    with open('data/vectorizer.pkl') as f:
        vectorizer = pickle.load(f)
    with open('data/model_response.pkl') as f:
        model_response = pickle.load(f)
    X = vectorizer.transform([text])
    predictions= model_response.predict(X)
    for prediction in predictions:
        if prediction == 0:
            return "Predict to close with NO RELIEF"
        if prediction == 1:
            return "Predcit to close with EXPLANATION"
        if prediction == 2:
            return "Predict to close with RELIEFs "

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
