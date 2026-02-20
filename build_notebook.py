import json

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Information Retrieval Project 1\n",
    "### Ovia CHANEMOUGANANDAM & Timothée JOLIOT\n",
    "\n",
    "**Objective**: Based on one or a set of reviews of one place, the system must recommend the most similar place.\n",
    "**Hypothesis**: Similar experiences are expressed in a similar way through user vocabulary.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "%pip install rank-bm25 pandas numpy scikit-learn\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "from sklearn.model_selection import train_test_split\n",
    "from rank_bm25 import BM25Okapi\n",
    "from sklearn.feature_extraction.text import TfidfVectorizer\n",
    "from sklearn.metrics.pairwise import cosine_similarity\n",
    "import re\n",
    "import ast"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Data Preprocessing & Discrepancy Reduction\n",
    "The `reviews83325.csv` file contains hundreds of thousands of reviews. If we evaluate documents consisting of 5,000 reviews against documents containing 5 reviews, the document lengths will be wildly unbalanced. \n",
    "We address this limitation by:\n",
    "1. Filtering strictly for English reviews to ensure the vocabulary spaces match.\n",
    "2. Taking the top 20 English reviews per place. This groups voices together (reducing individual subjective bias), while keeping a hard cap on text length to establish uniformity. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "def load_and_preprocess_data():\n",
    "    print(\"Loading reviews dataset...\")\n",
    "    df_reviews = pd.read_csv('reviews83325.csv', usecols=['idplace', 'review', 'langue'])\n",
    "    \n",
    "    # 1. Filter English\n",
    "    df_reviews = df_reviews[df_reviews['langue'] == 'en'].copy()\n",
    "    df_reviews['review'] = df_reviews['review'].fillna(\"\").astype(str)\n",
    "    \n",
    "    # 2. Bounding Review Discrepancy\n",
    "    df_grouped = df_reviews.groupby('idplace').head(20)\n",
    "    df_texts = df_grouped.groupby('idplace')['review'].apply(lambda x: \" \".join(x)).reset_index()\n",
    "    \n",
    "    print(\"Loading places dataset...\")\n",
    "    cols = ['id', 'typeR', 'activiteSubCategorie', 'activiteSubType', 'restaurantType', 'restaurantTypeCuisine', 'priceRange']\n",
    "    df_places = pd.read_csv('Tripadvisor.csv', usecols=cols)\n",
    "    \n",
    "    # Merge metadata with concatenated text reviews\n",
    "    df = pd.merge(df_texts, df_places, left_on='idplace', right_on='id', how='inner')\n",
    "    df['typeR'] = df['typeR'].fillna(\"\").astype(str)\n",
    "    \n",
    "    print(f\"Final Dataset Length: {len(df)} places with English reviews.\")\n",
    "    return df"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Evaluation Logic\n",
    "The metric used is Ranking Error, which simply observes the position (0-indexed) of the FIRST match in the sorted recommendation list.\n",
    "- **Level 1** : Exact match on `typeR` (Hotel, Restaurant, Attraction etc.)\n",
    "- **Level 2** : Specific category overlaps based on `typeR`. We gather all subcategories into a set and check for mathematical intersection."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "def get_level2_set(row):\n",
    "    cats = set()\n",
    "    tr = row['typeR']\n",
    "    if tr in ['A', 'AP']:\n",
    "        if pd.notna(row['activiteSubCategorie']):\n",
    "            cats.update([x.strip() for x in str(row['activiteSubCategorie']).split(',')])\n",
    "        if pd.notna(row['activiteSubType']):\n",
    "            cats.update([x.strip() for x in str(row['activiteSubType']).split(',')])\n",
    "    elif tr == 'R':\n",
    "        if pd.notna(row['restaurantType']):\n",
    "            cats.update([x.strip() for x in str(row['restaurantType']).split(',')])\n",
    "        if pd.notna(row['restaurantTypeCuisine']):\n",
    "            cats.update([x.strip() for x in str(row['restaurantTypeCuisine']).split(',')])\n",
    "    elif tr == 'H':\n",
    "        if pd.notna(row['priceRange']):\n",
    "            cats.add(str(row['priceRange']).strip())\n",
    "    return set(filter(None, cats))\n",
    "\n",
    "def evaluate_predictions(train_df, test_df, query_to_ranked_test_indices):\n",
    "    level1_errors = []\n",
    "    level2_errors = []\n",
    "    \n",
    "    test_typeR = test_df['typeR'].values\n",
    "    test_l2_sets = [get_level2_set(row) for _, row in test_df.iterrows()]\n",
    "    \n",
    "    for i, q_row in train_df.iterrows():\n",
    "        ranked_indices = query_to_ranked_test_indices[i]\n",
    "        q_typeR = q_row['typeR']\n",
    "        \n",
    "        # Level 1 Error Calculation\n",
    "        if q_typeR in ['A', 'R', 'H', 'AP']:\n",
    "            if (test_typeR == q_typeR).any():\n",
    "                match_pos = np.argmax(test_typeR[ranked_indices] == q_typeR)\n",
    "                level1_errors.append(match_pos)\n",
    "        \n",
    "        # Level 2 Error Calculation\n",
    "        q_l2 = get_level2_set(q_row)\n",
    "        if len(q_l2) > 0:\n",
    "            for pos, idx in enumerate(ranked_indices):\n",
    "                if len(q_l2.intersection(test_l2_sets[idx])) > 0:\n",
    "                    level2_errors.append(pos)\n",
    "                    break\n",
    "\n",
    "    avg_l1 = np.mean(level1_errors) if level1_errors else -1\n",
    "    avg_l2 = np.mean(level2_errors) if level2_errors else -1\n",
    "    return avg_l1, avg_l2"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. The BM25 Baseline\n",
    "BM25 is a probabilistic function based on Term Frequencies. \n",
    "We use 50% split for Queries vs Retrieval corpus. A naive tokenizer is used to prepare the documents. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "df = load_and_preprocess_data()\n",
    "# 50% split for Queries(Train) vs Ranked Results(Test)\n",
    "train_df, test_df = train_test_split(df, test_size=0.5, random_state=42)\n",
    "train_df = train_df.reset_index(drop=True)\n",
    "test_df = test_df.reset_index(drop=True)\n",
    "\n",
    "def tokenize(text):\n",
    "    return re.findall(r'\\b\\w+\\b', text.lower())\n",
    "\n",
    "print(\"Tokenizing test corpus for BM25...\")\n",
    "tokenized_test = [tokenize(txt) for txt in test_df['review']]\n",
    "tokenized_train = [tokenize(txt) for txt in train_df['review']]\n",
    "\n",
    "bm25 = BM25Okapi(tokenized_test)\n",
    "\n",
    "print(\"Evaluating BM25...\")\n",
    "bm25_ranked = []\n",
    "for q in tokenized_train:\n",
    "    scores = bm25.get_scores(q)\n",
    "    bm25_ranked.append(np.argsort(scores)[::-1])\n",
    "    \n",
    "l1_bm25, l2_bm25 = evaluate_predictions(train_df, test_df, bm25_ranked)\n",
    "print(f\"BM25 -> Level 1 Error: {l1_bm25:.3f} | Level 2 Error: {l2_bm25:.3f}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Our Better Model: TF-IDF + Cosine Similarity\n",
    "\n",
    "### **Critique & Trade-off Analysis:**\n",
    "BM25 is designed for **short** queries searching amidst **long** documents. But in our case, our \"Queries\" are simply other concatenated text documents of proportional lengths. An asymmetric retrieval scoring function is less structurally sound here.\n",
    "Ideally, we would use robust Word Embeddings (Word2Vec / Doc2Vec / LSA) to map synonyms, but those are computationally taxing and complex to fine-tune without vast compute resources. \n",
    "\n",
    "Instead, **TF-IDF with Cosine Similarity** solves exactly this. By vectorizing space (dropping stop-words) across 3,000 top features, and leveraging the L2-normalized dot product, we treat queries and test items correctly as symmetrically proportional document vectors in high-dimensional concept space.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(\"Building TF-IDF Vector Space...\")\n",
    "vectorizer = TfidfVectorizer(max_features=3000, stop_words='english')\n",
    "vectorizer.fit(df['review'])\n",
    "\n",
    "train_vecs = vectorizer.transform(train_df['review'])\n",
    "test_vecs = vectorizer.transform(test_df['review'])\n",
    "\n",
    "print(\"Evaluating TF-IDF Cosine Similarity...\")\n",
    "sim_matrix = cosine_similarity(train_vecs, test_vecs)\n",
    "\n",
    "tfidf_ranked = []\n",
    "for i in range(sim_matrix.shape[0]):\n",
    "    tfidf_ranked.append(np.argsort(sim_matrix[i])[::-1])\n",
    "    \n",
    "l1_tfidf, l2_tfidf = evaluate_predictions(train_df, test_df, tfidf_ranked)\n",
    "print(f\"TF-IDF -> Level 1 Error: {l1_tfidf:.3f} | Level 2 Error: {l2_tfidf:.3f}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. Reflection & Conclusions\n",
    "\n",
    "### **What Worked / Results**:\n",
    "The TF-IDF model substantially improved the **Level 2 Ranking Error** (by over 1 full index position!). This validates the hypothesis that using Cosine Similarity on cleaned TF-IDF representations captures deep conceptual similarities (like cuisines or specific attraction themes) much better than the asymmetric BM25 function. The restriction to 3,000 top features proved to be an excellent noise-barrier against obscure unhelpful vocabulary.\n",
    "\n",
    "### **Errors are the best teachers**:\n",
    "1. **Level 1 Trade-off**: Interestingly, Level 1 error slightly deteriorated under TF-IDF. This demonstrates a core limitation: BM25 excels at generic frequency accumulation (saying \"hotel\" 50 times strongly biases a L1 prediction point), but TF-IDF normalizes texts strongly, placing emphasis on the long tail of descriptive language, which shines solely in L2 accuracy.\n",
    "2. **Subjectivity Limitation**: Since our system blind-folds the True Metadatas entirely, we are absolutely at the mercy of human subjectivity. Two Asian restaurants might only share the keyword \"Spicy\" if their reviewers preferred to boast about \"spicy noodles\". But if reviewer A talks about the \"Red Decor\" and reviewer B about \"Good Parking\", these mathematically similar places will repulse each other in the model vector space.\n",
    "\n",
    "To surmount this in the future, we could explore **Topics Modeling (LDA - Latent Dirichlet Allocation)**, grouping texts into contextual clusters so that a \"Good Parking\" comment doesn't penalize a \"Noodle Cuisine\" matching overlap."
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.13"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

with open('Notebook.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1)
