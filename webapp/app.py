from flask import Flask,request,render_template
app = Flask(__name__)
import cPickle as pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
with open('data/vectorizer.pkl') as f:
    vectorizer = pickle.load(f)

with open('data/model_dispute.pkl') as f:
    model_dispute = pickle.load(f)

with open('data/model_response.pkl') as f:
    model_response = pickle.load(f)

def X_remove(text):
    chars_to_remove = ['XX', 'XXX', 'XXXX']
    text = text.translate(None, ''.join(chars_to_remove))
    return text
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
    text = X_remove(text)
    X = vectorizer.transform([text])
    pred_prob= model_dispute.predict_proba(X)[:,1]
    if pred_prob>=0.22:
        results = "This consumer is 'Likely to Dispute' and the probability for this prediction is: %5.3f" %pred_prob
    else:
        results = "This consumer is 'Not Likely to Dispute' and the probability for this prediction is: %5.3f" %pred_prob
    return render_template('result.html', prediction = results)

@app.route('/response_predict', methods=['POST'] )
def response_predict():
    text = str(request.form['consumer_input'])
    text = X_remove(text)
    X = vectorizer.transform([text])
    predictions= model_response.predict(X)
    pred_prob= model_response.predict_proba(X)[:,1]
    for prediction in predictions:
        if prediction == 0:
            results= "Your complaint is predicted to be 'closed with NO RELIEF' and the probability for this prediction is: %5.3f. You may want to take an extra look at your complaint" %pred_prob
        if prediction == 1:
            results =  "Your complaint is predicted to be 'closed with EXPLANATION' and the probability for this prediction is: %5.3f. It's OK since most of the people will get this type of response anyway." %pred_prob
        if prediction == 2:
            results =  "Congraduations! Your complaint is predicted to be 'closed with RELIEF' and the probability for this prediction is: %5.3f" %pred_prob
        return render_template('result.html', prediction = results)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
