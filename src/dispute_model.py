import pandas as pd
from sklearn.cross_validation import train_test_split
import sklearn.metrics as skm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier

def dispute_model(df):
    #subset the data for complaint narrative text that are not missing
    df=df[df['Consumer complaint narrative']!= 'None or Not Provided']
    #remove the 'XX','XXX','XXXX' characters that are used for filtering
    #personal info before it was published.
    chars_to_remove = ['XX', 'XXX', 'XXXX']
    df['Consumer complaint narrative'] = df['Consumer complaint narrative']  \
    .apply(lambda x: x.translate(None, ''.join(chars_to_remove)))

    X_word = df['Consumer complaint narrative'].values
    y = df['Consumer disputed?'].values
    #split datset into 80% training and 20% test
    X_train_word, X_test_word, y_train, y_test = train_test_split(X_word, y, \
    test_size=0.20, random_state=67)

    #tf-idf vector
    vectorizer = TfidfVectorizer(stop_words='english',lowercase=False,   \
    min_df=0.001, max_df = 0.2)
    #vectorization for train and test data,respectively
    words_matrix_train = vectorizer.fit_transform(X_train_word)
    words_matrix_test = vectorizer.transform(X_test_word)
    #SGD classifier using logistic loss function
    sgd = SGDClassifier(loss= 'log')
    sgd.fit(words_matrix_train, y_train)

    v_probs = sgd.predict_proba(words_matrix_test)[:, 1]

    return sgd, v_probs

def plot_roc(v_probs, y_test, title, xlabel, ylabel):
    # ROC
    fig = plt.figure(figsize = (8,6))
    tpr, fpr, thresholds = roc_curve(v_probs, y_test)

    auc = skm.roc_auc_score(y_test, v_probs)

    plt.hold(True)
    plt.plot(fpr, tpr)

    # 45 degree line
    xx = np.linspace(0, 1.0, 20)
    plt.plot(xx, xx, 'k--')

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title+auc)

    plt.show()

    return plot_roc(v_prob, y_test, "ROC plot of  complaint dispute prediction with text features,", 
             "False Positive Rate (1 - Specificity)", "True Positive Rate (Sensitivity, Recall)")
