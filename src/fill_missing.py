def fill_missing(df):
    # fill 'sub-issue' and 'Consumer complaint narrative' column missing value
    # with string value of 'Not Provided'
    df['Sub-issue'].fillna('Not Provided',inplace=True)
    df['Consumer complaint narrative'].fillna('None or Not Provided',inplace=True)
    # Combine "company public missing value" with "Company chose not to provide"
    df['Company public response'].fillna('Company chooses not to provide',inplace=True)

    #Combine missing value of "Issue" with "Other"
    df['Issue'].fillna('Other',inplace=True)

    # Replace missing vlaues of 'Tags' with "'Unknown'
    df['Tags'].fillna('Unknown',inplace=True)

    # Replace missing vlaues of 'Submitted via' with "'other'
    df['Submitted via'].fillna('Other',inplace=True)

    #Combine missing value,other,and withdrawn of "Consumer consent provided?" column
    #with Consumer consent not provided, since only users's complaints narrative will be provided
    # with the type of "Consumer consent provided"
    df['Consumer consent provided?'].fillna('Consent not provided',inplace=True)
    df['Consumer consent provided?']=df['Consumer consent provided?'].apply(lambda x:  \
            'Consent not provided' if x=='Other' or x=='Consent withdrawn' else x)

    return df
