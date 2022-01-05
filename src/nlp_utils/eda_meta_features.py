import ast
import pandas as pd
from collections import Counter

def words_unique(df, sentiment,numwords, raw_text, column):
    allother = [x for word in df[df.Label != sentiment][column] for x in ast.literal_eval(word)]

    allother  = list(set(allother))
    specific_only = [x for x in raw_text if x not in allother]
    
    counter = Counter([x for w in df[df.Label == sentiment][column] for x in ast.literal_eval(w)])
    keep = list(specific_only)
    
    for word in list(counter):
        if word not in keep:
            del counter[word]
    
    Unique_words = pd.DataFrame(counter.most_common(), columns = ['words','count'])
    
    return Unique_words