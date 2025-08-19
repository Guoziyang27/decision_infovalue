# 1. Setup and Imports
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import json
# Download NLTK data (if you haven't already)
nltk.download('stopwords')
nltk.download('wordnet')

# ---

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--user_label', type=int, default=1)
args = parser.parse_args()


# 2. Data Loading and Preprocessing
def preprocess_text(text):
    """Cleans and prepares a single text document."""
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I|re.A)
    # Convert to lowercase
    text = text.lower()
    # Tokenize (split into words)
    tokens = text.split()
    # Remove stopwords and lemmatize
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(lemmatized_tokens)

data = json.load(open('label.json'))
df = pd.DataFrame(data)

df = df.loc[df['user_label'] == args.user_label]

# Apply the preprocessing function to your text column
df['processed_text'] = df['review_text'].apply(preprocess_text)
print("--- Preprocessed Text ---")
print(df[['review_text', 'processed_text']].head())

# ---

# 3. Feature Extraction (Vectorization)
# Convert text data into numerical vectors using TF-IDF.
vectorizer = TfidfVectorizer(max_df=0.8, min_df=2, stop_words='english')
# `max_df=0.8`: ignore terms that appear in more than 80% of documents.
# `min_df=2`: ignore terms that appear in less than 2 documents.

X = vectorizer.fit_transform(df['processed_text'])

# Optional: Normalize the vectors to improve KMeans performance
X_normalized = normalize(X)

print("\n--- TF-IDF Matrix Shape ---")
print(f"Shape: {X.shape} (Documents, Features/Words)")

# ---

# 4. Clustering
# Define the number of clusters you want to find.
num_clusters = 20 

# Initialize and run the KMeans algorithm
km = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
km.fit(X_normalized)

# Assign the cluster labels to your DataFrame
df['cluster'] = km.labels_ + args.user_label * num_clusters
print("\n--- Data with Cluster Assignments ---")
print(df[['review_text', 'cluster']].head(10))

# ---

# 5. Analysis and Interpretation
# To understand the clusters, you can find the most common words in each one.
print("\n--- Top terms per cluster ---")
order_centroids = km.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names_out()

for i in range(num_clusters):
    print(f"Cluster {i}:", end="")
    for ind in order_centroids[i, :10]: # Get top 10 terms
        print(f" {terms[ind]}", end="")
    print()

df.to_csv(f'hotel_review_user_label_{args.user_label}.csv', index=False)