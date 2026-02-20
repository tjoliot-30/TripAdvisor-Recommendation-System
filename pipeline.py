import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import ast

def process_data():
    print("Loading datasets...")
    # Load reviews
    df_reviews = pd.read_csv('reviews83325.csv', usecols=['idplace', 'review', 'langue'])
    # Filter English
    df_reviews = df_reviews[df_reviews['langue'] == 'en']
    df_reviews['review'] = df_reviews['review'].fillna("").astype(str)
    
    # Take up to 20 reviews per place to reduce disparity and memory
    df_grouped = df_reviews.groupby('idplace').head(20)
    # Concatenate reviews for each place
    df_texts = df_grouped.groupby('idplace')['review'].apply(lambda x: " ".join(x)).reset_index()
    
    # Load places
    cols = ['id', 'typeR', 'activiteSubCategorie', 'activiteSubType', 'restaurantType', 'restaurantTypeCuisine', 'priceRange']
    df_places = pd.read_csv('Tripadvisor.csv', usecols=cols)
    
    # Join
    df = pd.merge(df_texts, df_places, left_on='idplace', right_on='id', how='inner')
    
    # Some basic cleaning
    df['typeR'] = df['typeR'].fillna("").astype(str)
    print(f"Total places with english reviews: {len(df)}")
    
    return df

def get_level2_set(row):
    cats = set()
    tr = row['typeR']
    if tr in ['A', 'AP']:
        if pd.notna(row['activiteSubCategorie']):
            cats.update([x.strip() for x in str(row['activiteSubCategorie']).split(',')])
        if pd.notna(row['activiteSubType']):
            cats.update([x.strip() for x in str(row['activiteSubType']).split(',')])
    elif tr == 'R':
        if pd.notna(row['restaurantType']):
            cats.update([x.strip() for x in str(row['restaurantType']).split(',')])
        if pd.notna(row['restaurantTypeCuisine']):
            cats.update([x.strip() for x in str(row['restaurantTypeCuisine']).split(',')])
    elif tr == 'H':
        if pd.notna(row['priceRange']):
            cats.add(str(row['priceRange']).strip())
    return cats

def evaluate_predictions(train_df, test_df, query_to_ranked_test_indices):
    level1_errors = []
    level2_errors = []
    
    test_typeR = test_df['typeR'].values
    # Precompute valid types and level 2 sets for the test items for fast lookup
    test_l2_sets = [get_level2_set(row) for _, row in test_df.iterrows()]
    
    for i, q_row in train_df.iterrows():
        ranked_indices = query_to_ranked_test_indices[i]
        
        # Level 1 evaluation
        q_typeR = q_row['typeR']
        if q_typeR in ['A', 'R', 'H', 'AP']:
            # Has an answer in test ?
            if (test_typeR == q_typeR).any():
                match_pos = np.argmax(test_typeR[ranked_indices] == q_typeR)
                level1_errors.append(match_pos)
        
        # Level 2 evaluation
        q_l2 = get_level2_set(q_row)
        if len(q_l2) > 0:
            # Check if any test item has intersection
            has_match = False
            for idx in ranked_indices:
                if len(q_l2.intersection(test_l2_sets[idx])) > 0:
                    has_match = True
                    break
            
            if has_match:
                # Find position
                for pos, idx in enumerate(ranked_indices):
                    if len(q_l2.intersection(test_l2_sets[idx])) > 0:
                        level2_errors.append(pos)
                        break

    avg_l1 = np.mean(level1_errors) if level1_errors else -1
    avg_l2 = np.mean(level2_errors) if level2_errors else -1
    return avg_l1, avg_l2

def run():
    df = process_data()
    train_df, test_df = train_test_split(df, test_size=0.5, random_state=42)
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    
    # Basic tokenization
    def tokenize(text):
        return re.findall(r'\b\w+\b', text.lower())

    print("Tokenizing for BM25...")
    tokenized_test = [tokenize(txt) for txt in test_df['review']]
    tokenized_train = [tokenize(txt) for txt in train_df['review']]
    
    print("Running BM25...")
    bm25 = BM25Okapi(tokenized_test)
    
    bm25_ranked = []
    for q in tokenized_train:
        scores = bm25.get_scores(q)
        ranked = np.argsort(scores)[::-1]
        bm25_ranked.append(ranked)
        
    l1_bm25, l2_bm25 = evaluate_predictions(train_df, test_df, bm25_ranked)
    print(f"BM25 -> Level 1 Error: {l1_bm25:.2f}, Level 2 Error: {l2_bm25:.2f}")
    
    print("Running TF-IDF (Better model)...")
    vectorizer = TfidfVectorizer(max_features=3000, stop_words='english')
    # Fit on all to have a good vocabulary
    vectorizer.fit(df['review'])
    train_vecs = vectorizer.transform(train_df['review'])
    test_vecs = vectorizer.transform(test_df['review'])
    
    sim_matrix = cosine_similarity(train_vecs, test_vecs)
    tfidf_ranked = []
    for i in range(sim_matrix.shape[0]):
        ranked = np.argsort(sim_matrix[i])[::-1]
        tfidf_ranked.append(ranked)
        
    l1_tfidf, l2_tfidf = evaluate_predictions(train_df, test_df, tfidf_ranked)
    print(f"TF-IDF -> Level 1 Error: {l1_tfidf:.2f}, Level 2 Error: {l2_tfidf:.2f}")

    with open('results.txt', 'w') as f:
        f.write(f"BM25 L1: {l1_bm25}\nBM25 L2: {l2_bm25}\nTFIDF L1: {l1_tfidf}\nTFIDF L2: {l2_tfidf}")

if __name__ == '__main__':
    run()
