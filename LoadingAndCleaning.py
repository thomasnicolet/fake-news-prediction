# based on 
# https://towardsdatascience.com/predicting-fake-news-using-nlp-and-machine-learning-scikit-learn-glove-keras-lstm-7bbd557c3443
import pandas as pd
import numpy as np
import contractions 
import re
import nltk
import time 
from multiprocessing import Pool

#############
# TODOS:
# (1) Clean and document
#############

def trim_df (df):

     #df = df[["domain", "type", "url", "content", "title", "authors", "scraped_at"]]
    df['raw_content_length'] = df['content'].apply(lambda x: len(x))
    df = df[df['raw_content_length'] < 24000]

    df = df[["domain", "type", "url", "content", "title", "authors", "scraped_at"]]
    df = df[df.type != "unknown"]
    
    # TODO: Fix this warning 
    df['label'] = np.where(((df['type'] == 'political') | (df['type'] == 'reliable')), 1, 0)
    df = df.dropna(subset=['type', 'content'])
    df['content'] = df['content'].str.strip()
    df = df.drop_duplicates(subset="content") 
    df = df.fillna('')

    return df

def trim_df_only_content (df):
   
    
    df['raw_content_length'] = df['content'].apply(lambda x: len(x))
    df = df[df['raw_content_length'] < 24000]
    df = df[df.type != "unknown"]
    df = df[["type","content", ]]
    # TODO: Fix this warning 
    df['label'] = np.where(((df['type'] == 'political') | (df['type'] == 'reliable')), 1, 0)
    df = df.dropna(subset=['type', 'content'])
    df['content'] = df['content'].str.strip()
    df = df.drop_duplicates(subset="content") 
    df = df.fillna('')

    return df

def clean_content(series):
    import re
    regex_oddcharacters = r'[^\d\w\s]'
    regex_whitespaces   = r'\s\s+'
    regex_date          = r'\d{4}-\d{2}-\d{2} \d{2}\:\d{2}\:\d{2}\.\d*'
    regex_email         = r'[^@ \t\r\n]+@[^@ \t\r\n]+\.[^@ \t\r\n]+'
    regex_url           = r'((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*'
    regex_num           = r'(?:\+|\-|\$)?\d{1,}(?:[\.\,]?\d+)*(?:[\.\,]\d+)?%?(th|rd|st|nd)?'
    def clean_text(text):
        '''
        Cleans given input string. Lowers text, removes leading and trailing whitespaces,
        removes multiple whitespaces, tabs and newlines.
        Replaces dates, emails, urls and nums with e.g. <DATE>
        '''
        # Note: Order is important
        text = text.lower()
        text = contractions.fix(text)
        text = text.strip()
        text = re.sub(regex_whitespaces, " ", text)
        text = re.sub(regex_date, 'DATEDATE', text)
        text = re.sub(regex_email, 'EMAILEMAIL', text)
        text = re.sub(regex_url, 'URLURL', text)
        text = re.sub(regex_num, 'NUMNUM', text)
        text = re.sub(regex_oddcharacters, "", text)

        return text
    
    series = series.apply(clean_text)
    
    return series


def clean_content_without_substitution(series):
    import re
    regex_oddcharacters = r'[^\d\w\s]'
    regex_whitespaces   = r'\s\s+'
    regex_date          = r'\d{4}-\d{2}-\d{2} \d{2}\:\d{2}\:\d{2}\.\d*'
    regex_email         = r'[^@ \t\r\n]+@[^@ \t\r\n]+\.[^@ \t\r\n]+'
    regex_url           = r'((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*'
    regex_num           = r'(?:\+|\-|\$)?\d{1,}(?:[\.\,]?\d+)*(?:[\.\,]\d+)?%?(th|rd|st|nd)?'
    def clean_text(text):
        '''
        Cleans given input string. Lowers text, removes leading and trailing whitespaces,
        removes multiple whitespaces, tabs and newlines.
        Replaces dates, emails, urls and nums with e.g. <DATE>
        '''
        # Note: Order is important
        text = text.lower()
        text = contractions.fix(text)
        text = text.strip()
        text = re.sub(regex_whitespaces, " ", text)
        text = re.sub(regex_oddcharacters, "", text)

        return text
    
    series = series.apply(clean_text)
    
    return series


def drop_empty_string_rows(df):
    '''
    Drops any row if it is just an empty string
    Note these used to be nan's
    '''
    for col in df:
        df = df.loc[df[col] != '']
    return df
def tokenize(series):
    series = series.apply(nltk.word_tokenize)
    return series

def remove_stopwords(series):
    stopwords = set(nltk.corpus.stopwords.words('english'))
    series = series.apply(lambda tokens: [t for t in tokens if t not in stopwords])
    return series

def recombine_content(series):
    # We want to use count vectorizer which expects strings, so join the token list back into string
    series = series.apply(lambda x: " ".join(x))
    return series

def add_content_length(series):
    series = series.apply(lambda x: len(x))
    return series

def clean_authors(series):
    regex_oddcharacters = r'[^\d\w\s]'
    series = series.apply(lambda authorstring: authorstring.split(",") if type(authorstring) is str else [])
    series = series.apply(lambda authorlist: [author.strip() for author in authorlist])
    series = series.apply(lambda authorlist: [author.lower() for author in authorlist])
    series = series.apply(lambda authorlist: [(re.sub(regex_oddcharacters, "", author)) for author in authorlist])
    return series

# If we want to collapse author list and append to content
def join_author_list_to_string(df):
    for idx in df.index: 
        author_lst = df['authors'][idx]
        author_str = ''.join(author_lst)
        df['authors'][idx] = author_str
    return df


def init_dataframe(load_path = None, percent_load=100):

    import random
    start_time = time.time() 
    print("Initializing dataframe...")

    df = pd.read_csv(
            load_path,
            dtype=np.string_,
            skiprows=lambda i: i>0 and random.random() > percent_load
            )    

    end_time = time.time()
    print(f"Initialized data ({df.shape[0]} entries) in: {end_time-start_time:0.1f} seconds") 
    
    return df

def clean_df(df, use_multiprocessing, do_benchmark=True, is_clean_authors_and_title=False, skip_substitution=False):

    #print("Trimming entries...")
    start_time = time.time() 

    #df = trim_df(df, only_content)
    # Notice we separated dropping nan by first making empty strings inplace
    # then we drop empty strings (remove if we want to keep nan cols)
    # dropping nan makes even genuine/fake articles, but if we keep nan cols
    # we can randomly distribute genuine/fake articles ourselves
    # if do_benchmark:
    #     read_time = time.time()
    #     print(f"Trimmed ({df.shape[0]} entries) in: {read_time-start_time:0.1f} seconds") 

    print("Cleaning...")
    

    if skip_substitution: # this is pretty stupid, just checking difference
        print("Not substituting!")
        import multiprocessing as mp
        with mp.Pool() as pool:
            df["title"] = pd.concat(pool.map(clean_content_without_substitution, np.array_split(df["title"], 32)))
            df["authors"] = pd.concat(pool.map(clean_content_without_substitution, np.array_split(df["authors"], 32)))
            df["content"] = pd.concat(pool.map(clean_content_without_substitution, np.array_split(df["content"], 32)))
            processed_time = time.time()
            print(f"Cleaned ({df.shape[0]} entries) in: {processed_time-start_time:0.1f} seconds")
            return df
        
    elif use_multiprocessing:
        import multiprocessing as mp
        if is_clean_authors_and_title:
            with mp.Pool() as pool:
                
                df["title"] = pd.concat(pool.map(clean_content, np.array_split(df["title"], 32)))
                df["authors"] = pd.concat(pool.map(clean_content, np.array_split(df["authors"], 32)))
                df["content"] = pd.concat(pool.map(clean_content, np.array_split(df["content"], 32)))
                processed_time = time.time()
                print(f"Cleaned ({df.shape[0]} entries) in: {processed_time-start_time:0.1f} seconds")
                return df
                
        else:
            with mp.Pool() as pool:
                df["content"] = pd.concat(pool.map(clean_content, np.array_split(df["content"], 32)))
                processed_time = time.time()
                print(f"Cleaned ({df.shape[0]} entries) in: {processed_time-start_time:0.1f} seconds")
                return df
    else: 
        if is_clean_authors_and_title:
            df["content"] = pd.concat(map(clean_content, np.array_split(df["content"], 32)))
            df["title"]   = pd.concat(map(clean_content, np.array_split(df["title"], 32)))
            df["authors"] = pd.concat(map(clean_content, np.array_split(df["authors"], 32)))
            processed_time = time.time()
            print(f"Cleaned ({df.shape[0]} entries) in: {processed_time-start_time:0.1f} seconds")
            return df
        else: 
            df["content"] = pd.concat(map(clean_content, np.array_split(df["content"], 32)))
            processed_time = time.time()
            print(f"Cleaned ({df.shape[0]} entries) in: {processed_time-start_time:0.1f} seconds")
            return df



def process_df(df, use_multiprocessing=False):
    start_time = time.time() 

    print("Processing...")
    if use_multiprocessing:
        import multiprocessing as mp
       
        with mp.Pool() as pool:
            df["content"] = pd.concat(pool.map(tokenize, np.array_split(df["content"], 32)))
            df["content"] = pd.concat(pool.map(remove_stopwords, np.array_split(df["content"], 32)))
            
            df["title"] = pd.concat(pool.map(tokenize, np.array_split(df["title"], 32)))
            df["title"] = pd.concat(pool.map(remove_stopwords, np.array_split(df["title"], 32)))
            
            df["authors"] = pd.concat(pool.map(tokenize, np.array_split(df["authors"], 32)))
            df["authors"] = pd.concat(pool.map(remove_stopwords, np.array_split(df["authors"], 32)))

            df["content_joined"] = pd.concat(pool.map(recombine_content, np.array_split(df["content"], 32)))
            df["content_length"] = pd.concat(pool.map(add_content_length, np.array_split(df["content"], 32)))

    else:
        df["content"] = pd.concat(map(tokenize, np.array_split(df["content"], 32)))
        df["content"] = pd.concat(map(remove_stopwords, np.array_split(df["content"], 32)))
        
        df["title"] = pd.concat(map(tokenize, np.array_split(df["title"], 32)))
        df["title"] = pd.concat(map(remove_stopwords, np.array_split(df["title"], 32)))
        
        df["authors"] = pd.concat(map(tokenize, np.array_split(df["authors"], 32)))
        df["authors"] = pd.concat(map(remove_stopwords, np.array_split(df["authors"], 32)))

        df["content_joined"] = pd.concat(map(recombine_content, np.array_split(df["content"], 32)))
        df["content_length"] = pd.concat(map(add_content_length, np.array_split(df["content"], 32)))
    processed_time = time.time()
    print(f"Processed({df.shape[0]} entries) in: {processed_time-start_time:0.1f} seconds")
    
    return df

def process_df_only_content(df, use_multiprocessing=False):
    start_time = time.time() 

    print("Processing...")
    if use_multiprocessing:
        import multiprocessing as mp
        print("Im using multiprocessing :)")
       
        with mp.Pool() as pool:
            df["content"] = pd.concat(pool.map(tokenize, np.array_split(df["content"], 32)))
            df["content"] = pd.concat(pool.map(remove_stopwords, np.array_split(df["content"], 32)))

            df["content_joined"] = pd.concat(pool.map(recombine_content, np.array_split(df["content"], 32)))
            df["content_length"] = pd.concat(pool.map(add_content_length, np.array_split(df["content"], 32)))

    else:
        df["content"] = pd.concat(map(tokenize, np.array_split(df["content"], 32)))
        df["content"] = pd.concat(map(remove_stopwords, np.array_split(df["content"], 32)))

        df["content_joined"] = pd.concat(map(recombine_content, np.array_split(df["content"], 32)))
        df["content_length"] = pd.concat(map(add_content_length, np.array_split(df["content"], 32)))

    processed_time = time.time()
    print(f"Processed({df.shape[0]} entries) in: {processed_time-start_time:0.1f} seconds")
    
    return df




def clean_df_only_content(df, do_benchmark=True):

    start_time = time.time() 
    print("Cleaning...")

    df["content"] = pd.concat(map(clean_content, np.array_split(df["content"], 32)))

    if do_benchmark:
        processed_time = time.time()
        print(f"Cleaned ({df.shape[0]} entries) in: {processed_time-start_time:0.1f} seconds")
  
    return df