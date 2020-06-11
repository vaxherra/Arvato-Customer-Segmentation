import numpy as np
import pandas as pd

import numpy as np
import pandas as pd

def unknown_to_nan(df, xlsx, missing_keyword = 'unknown', rename_columns = None ):
    """
    A helper function to convert missing entries to NaNs. 
    
    Using target dataset in the form of pandas dataframe (df) and provided metadata in 'xlsx' format 
    converts entries in `Values` column matching to provided 'missing_keyword' in the 'Meaning' column of the metadata file (xlsx).
    
    Returns an copy of the input data dataframe with replaced values.
    
    Args:
        df: input dataset (pandas.DataFrame) containing a list of features in columns
        xlsx: a path to input Excel metadata file
        missing_keyword: a string containing a keyword that indicates a missing/unknown/NaN value in the metadata file, defaults to 'unknown'
 
    """
    # Load mapping in xlsx format
    xlsx = pd.read_excel(xlsx, header = 1)
    #extra column called Unnamed that seems like an index duplication, I will drop it
    xlsx.drop(columns=['Unnamed: 0'], inplace=True)
    xlsx['Attribute'] = xlsx['Attribute'].ffill()
    xlsx['Description'] = xlsx['Description'].ffill()
    
    if(rename_columns is not None):
        print("Renaming columns according to provided file: {}".format(rename_columns))
    
        corrected_features = pd.read_csv(rename_columns,sep="\t").dropna()

        try:
            matching_columns =  sum([ 1 for column in  list(corrected_features.columns  ) if column in ['data','metadata'] ])
            assert( matching_columns==2)
        except:
            raise Exception("Provided metadata correction file {} lacks 2 requited columns: 'data', 'metadata'")

        corrected_features_dict = dict(zip(corrected_features['data'], corrected_features['metadata']))
        #rename columns
        df.rename(columns=corrected_features_dict, inplace=True)

    
    # First, obtain table with unknown feature values: a subset of provided xlsx file containing 'missing_keyword' keyword in the 'Meaning' field
    
    #using the DIAs xls file lets save meanings that might indicate unknown values
    unknowns = xlsx['Meaning'].where(xlsx['Meaning'].str.contains(missing_keyword)).value_counts().index
    missing_unknowns = xlsx[xlsx['Meaning'].isin(unknowns)]
    print("Found {} unique combination of values indicating/encoding missing entries \n".format( missing_unknowns['Value'].unique()  ))
    
    # First ensure that all columns (features) in df have metadata entry:
    missing_metadata_annotations = []
    for feature in list(df.columns):
        if(feature not in set(xlsx['Attribute']) ):
            msg = "{} feature is missing metadata annotation!".format(feature)
            #raise Exception(msg)
            missing_metadata_annotations.append(feature)
            print("Warning! " + msg)
    
    # Iterate through unknown features and its values
    not_present_features = []
    
    for attribute in set(xlsx['Attribute']):
        if(attribute not in set(df.columns)):
            print("Info!: {} feature in metadata not present in target dataset".format(attribute))
            not_present_features.append(attribute)
            

    
    for row in missing_unknowns.iterrows():
        row_idx = row[0] # row index
        row_vals = row[1] # columns

        attribute = row_vals['Attribute'] # attribute name
        
        # the values in the 'Value' column can be integer, like -1, or  a string "-1,0".
        # we want to convert them temporarily to a list of integers split by coma ',': i.e. -1 -> [-1], and "-1,0" -> [-1,0]
        # this allows ease of interpretation
        missing_values = row_vals['Value']
        missing_values = [ int(x) for x in str(missing_values).split(',')] # categories

        # An OPTIONAL check. If a given metadata is not present in target dataset (df), produce a warning.
        #    lots of missing values might indicate a problem with RAW dataset: (not expected)
        if(attribute not in set(df.columns)):
            continue # skipping, if a missing annotation for a metadata is not present in target dataset

            
        # Loop over missing_values, match and replace target dataset
        for missing_value in missing_values:
            # match column name (feature), and replace a given missing value with NAN
            df[attribute].replace(missing_values, np.nan, inplace=True)

    return df,list(set(missing_metadata_annotations)), list(set(not_present_features))
