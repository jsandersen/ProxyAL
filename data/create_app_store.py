from sklearn.utils import shuffle
import pandas as pd
import re
import numpy as np
import os

def _load_app_store(data_dir = './datasets/dataset.csv'):
    df = pd.read_csv(data_dir)
    
    df_en = df[(df.lang == 'en') & (df.source == 'app_review') & ~df['text'].isna()]
    
    X = df_en['text'].to_list()
    y = df_en['category'].map({'inq': 0, 'pbr': 1, 'irr': 2}).to_list()
    
    return X, y

# to remove URLs
def remove_URL(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'',text)

# to remove html tags
def remove_html(text):
    html=re.compile(r'<.*?>')
    return html.sub(r'',text)

if __name__ == "__main__":
    print('Prepare AppStore dataset ...')
    
    if not (os.path.isfile('./datasets/app_store_X.npy') and os.path.isfile('./datasets/app_store_X.npy')): 
        X, y = _load_app_store()
        X, y = shuffle(X, y, random_state = 42)
        
        df = pd.DataFrame(data=(X))
        df[0] = df[0].apply(remove_URL)
        df[0] = df[0].apply(remove_html)

        
        
        np.save('./datasets/app_store_X', df[0].tolist())
        np.save('./datasets/app_store_y', y)
        #np.save('imdb_index', [i for i in range(len(X))])
        
        
        print('Done')
    else:
        print('AppStore dataset already exists.')
    