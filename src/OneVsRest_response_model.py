from sklearn.preprocessing import StandardScaler
def OneVsRest_reponse_model(df):

    X = df_model.values
    y = df['Company response to consumer']
    #binarize the label for one vs rest classifier
    y_bin = label_binarize(y, classes=[0, 1, 2])
    #split datset into 80% training and 20% test
    X_train, X_test, y_train_bin, y_test_bin = train_test_split(X, y_bin, test_size=0.20, random_state=67)
    #normalize the feature in scale
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # Learn to predict each class against the other using random forest classifier
    classifier = OneVsRestClassifier(RandomForestClassifier(n_estimators=500, n_jobs=-1, class_weight='auto' ), n_jobs=-1)
    y_score = classifier.fit(X_train, y_train_bin).predict_proba(X_test)

    return y_score, y_test_bin

def plot_OneVsRest_roc(y_score,n_classes=3, y_test)
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i],y_score[:, i],drop_intermediate=False)
        roc_auc[i] = auc(fpr[i], tpr[i])


    fig = plt.figure(figsize=(8,6))
    lable = ['closed without relief', 'closed with explaination', 'closed with relief']
    for i,v in enumerate(lable):
        plt.plot(fpr[i], tpr[i], label= v + ' (auc_area = {1:0.2f})'
                                       ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC to company response prediction via non-complaint text features')
    plt.legend(loc="lower right")
    font = {'family' : 'normal',
            'weight' : 'bold',
            'size'   : 22}
    plt.rcParams.update({'font.size': 12})

    plt.show()

    return plot_OneVsRest_roc(y_score,n_classes=3, y_test_bin)
    
