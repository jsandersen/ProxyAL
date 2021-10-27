from sklearn.utils import shuffle
import pandas as pd
import re
import numpy as np
import os

def _load_imdb(data_dir = '/home/jovyan/Active Learning/data/datasets/aclImdb/'):
    X = []
    y = []
    for partition in ["train", "test"]:
        for category  in ["pos", "neg"]:
            lable = 0 if category  == "neg" else 1

            path = os.path.join(data_dir, partition, category )
            files = os.listdir(path)
            for f_name in files:
                with open(os.path.join(path, f_name), "r") as f:
                    review = f.read()
                    X.append(review)
                    y.append(lable)

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
    print('Prepare IMDB dataset ...')
    
    if not (os.path.isfile('./datasets/imdb_X.npy') and os.path.isfile('./datasets/imdb_X.npy')): 
        X, y = _load_imdb()
        X, y = shuffle(X, y, random_state = 42)
        
        df = pd.DataFrame(data=(X))
        df[0] = df[0].apply(remove_URL)
        df[0] = df[0].apply(remove_html)

        np.save('./datasets/imdb_X', df[0].tolist())
        np.save('./datasets/imdb_y', y)
        #np.save('imdb_index', [i for i in range(len(X))])
        
        print('Done')
    else:
        print('IMDB dataset already exists.')
    