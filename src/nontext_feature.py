import pandas as pd
from pandas import DataFrame
import datetime

def nontext_feature(df):
    df_model = DataFrame() #Creat a new dataframe for all the features in the model

    #Assign category feature values into categorical numerical value so as the
    #ML modle fitting can handle it
    feature_for_model=['Product', 'Sub-product','Issue','Sub-issue','Tags', 'State']
    for name in feature_for_model:
        repl={}
        i=0
        for value in df[name].unique():
            repl[value] = i
            i+=1

    df[name] = df[name].apply(lambda x: repl[x])
    df_model[name] = df[name].astype('category')


    #Creat feature for 'Timely response' boolean
    replace = {'Yes': True, 'No': False}
    df['Timely response?'] = df['Timely response?'].apply(lambda x: replace[x])
    df_model['Timely response?']=df['Timely response?']


    #Creat feature for 'Consumer consent provided?' boolean
    replace1={'Consent provided':True, 'Consent not provided':False}
    df_model['Consumer consent provided?']= df['Consumer consent provided?'] \
    .apply(lambda x: replace1[x])


    ##process time refers to days between the date CFPB received complaitns and the date
    ##when complaints were sent to company on behal of comsumer
    df['Date received']=pd.DatetimeIndex(df['Date received'],format='%m/%d/%Y').date
    df['Date sent to company']=pd.DatetimeIndex(df['Date sent to company'],format='%m/%d/%Y').date

    df['Process time']=(df['Date sent to company']-df['Date received']).astype('timedelta64[D]').astype(int)

    ##Create features about complaint submitted time
    df_model['Date_received_year'] = df['Date received'].apply(lambda x: x.year)
    df_model['Date_sent_month'] = df['Date sent to company'].apply(lambda x: x.month)
    df_model['Date_sent_day'] = df['Date sent to company'].apply(lambda x: x.day)



    #count the number of complaints for each company
    company_complaitns_counts = df['Company'].value_counts()
    df_model['company_complaint_counts'] = df['Company']  \
    .apply(lambda x: company_complaitns_counts[x])

    return df_model
