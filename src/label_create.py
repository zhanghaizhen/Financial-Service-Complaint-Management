# creating label for dispute prediction
import pandas as pd

def dispute_label(df):
    replace1={'Yes':True, 'No':False}

    df['Consumer disputed?']= df['Consumer disputed?'].apply(lambda x: replace1[x])

    return df

# creating label for company response prediction
def response_label(df):
    #remove the complaints with a response status as "In progress"
    df = df[df['Company response to consumer']!='In progress']
    #classify company's response into three categories
    replace2={'Closed':0, 'Untimely response':0,'Closed without relief':0, \
    'Closed with explanation':1, 'Closed with non-monetary relief':2,     \
    'Closed with relief': 2, 'Closed with monetary relief':2}

    df['Company response to consumer']= df['Company response to consumer'] \
        .apply(lambda x: replace2[x])

    return df
