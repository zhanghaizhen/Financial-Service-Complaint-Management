
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt

def response_model(df, df_model):
    X = df_model.values
    y = df['Company response to consumer'].values
    #split datset into 80% training and 20% test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=67)
    #normalize the feature in scale
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    #random forest classifier
    rfc = RandomForestClassifier(n_estimators=500, n_jobs=-1, class_weight='auto' )
    rfc.fit(X_train, y_train)
    #accuracy of test dataset
    rfc.score(X_test, y_test)

    return rfc

def plot_importance(rfc, df_model, max_features=10):
    '''Plot feature importance'''
    feature_importance = clf.feature_importances_
    # make importances relative to max importance
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + .5

    # Show only top features
    pos = pos[-max_features:]
    feature_importance = (feature_importance[sorted_idx])[-max_features:]
    feature_names = (X.columns[sorted_idx])[-max_features:]

    plt.barh(pos, feature_importance, align='center')
    plt.yticks(pos, feature_names)
    plt.xlabel('Relative Importance')
    plt.title('Non-Text Feature Importance')
    font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}
    plt.rcParams.update({'font.size': 16})

    plt.show()

    return plot_importance(rfc, df_model, max_features=20)
