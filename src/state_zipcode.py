# Fill missing 'State' info using valid zipcode.
import pandas as pd
from pyzipcode import ZipCodeDatabase

def zipcode_state(df):
    from pyzipcode import ZipCodeDatabase
    zip=ZipCodeDatabase()
    for i in df[pd.isnull(df['State'])&pd.notnull(df['ZIP code'])].index:
    try:
        df['State'][i]=str(zip[df['ZIP code'][i]].state)
    except:
        continue

    # fill the rest of the missing values of state and zipcode with string "Not provided"
    df['State'].fillna('Not provided',inplace=True)
    df['ZIP code'].fillna('Not Provided',inplace=True)

    return df
