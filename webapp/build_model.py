import cPickle as pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import train_test_split



def build_model(filename):
    #read in data with complaints text and finished the process
    complaints = pd.read_csv(filename)
    complaints = complaints.dropna(subset=['Consumer complaint narrative'])
    complaints = complaints[complaints['Company response to consumer']!='In progress']

    #build tfidf vectorizer on "Consumer complaint narrative"
    tfidf_vect = TfidfVectorizer(stop_words='english',lowercase=False, \
                min_df=0.001, max_df = 0.2)
    narrative = complaints['Consumer complaint narrative'].values
    tfidf = tfidf_vect.fit_transform(narrative) #build tfidf from complaint text
    X = tfidf

    #creat boolean lables for consumer dispute prediction
    replace1={'Yes':True, 'No':False}
    complaints['Consumer disputed?']= complaints['Consumer disputed?'] \
                .apply(lambda x: replace1[x])
    y1 = complaints['Consumer disputed?'].values

    #create categorical lables for company response to consumer prediction
    replace2={'Closed':0, 'Untimely response':0,'Closed without relief':0, \
                'Closed with explanation':1, 'Closed with non-monetary relief':2,  \
                    'Closed with relief': 2, 'Closed with monetary relief':2}
    complaints['Company response to consumer']= complaints['Company response to consumer'] \
            .apply(lambda x: replace2[x])
    y2 = complaints['Company response to consumer'].values

    #create models for dispute prediction and company response prediction, respectvely.
    clf_dispute = MultinomialNB()
    clf_response = MultinomialNB()

    vectorizer = tfidf_vect
    model_dispute = clf_dispute.fit(X, y1)
    model_response = clf_response.fit(X, y2)

    return vectorizer, model_dispute, model_response


if __name__ == '__main__':
    vectorizer, model_dispute, model_response = build_model('data/Consumer_complaints.csv')
    with open('data/vectorizer.pkl', 'w') as f:
        pickle.dump(vectorizer, f)
    with open('data/model_dispute.pkl', 'w') as f:
        pickle.dump(model_dispute, f)
    with open('data/model_response.pkl', 'w') as f:
        pickle.dump(model_response, f)
