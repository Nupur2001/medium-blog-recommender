import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# cosine_similarity: the tool that measures the "angle" between two vectors
# — this is what actually finds "closeness" between articles

# --- Same setup as vectorize.py ---
df = pd.read_csv("medium_articles.csv")

tech_keywords = ["Technology", "Artificial Intelligence", "Machine Learning",
                  "Data Science", "Programming", "Tech", "Software", "AI"]
mask = df["tags"].str.contains("|".join(tech_keywords), case=False, na=False)
df_sample = df[mask].sample(40, random_state=42).reset_index(drop=True)
df_sample["combined_text"] = df_sample["title"].fillna("") + " " + df_sample["text"].fillna("")

vectorizer = TfidfVectorizer(stop_words="english", max_features=500)
tfidf_matrix = vectorizer.fit_transform(df_sample["combined_text"])
# tfidf_matrix now holds the fingerprints for all 40 existing articles

# --- NEW PART: your own unpolished draft ---
my_draft = """
I've been experimenting with local LLMs on my Mac and I'm honestly surprised
how well a quantized 8B model runs. Setting up Ollama took five minutes and
now I can prompt a model completely offline. It feels like a big shift for
developers who want privacy and don't want to pay per API call.
"""
# Replace this with YOUR actual draft text when testing

draft_vector = vectorizer.transform([my_draft])
# IMPORTANT: we use .transform() here, NOT .fit_transform()
# .fit() already happened on the 40 articles — that's what LEARNED the vocabulary
# We must reuse that SAME vocabulary for the draft, otherwise we'd be comparing
# fingerprints built from two different rulers — meaningless comparison
# .transform() takes a LIST of documents, so we wrap my_draft in [ ]

# --- Compare the draft against all 40 articles at once ---
similarity_scores = cosine_similarity(draft_vector, tfidf_matrix)
# Returns a grid of similarity scores: 1 row (your draft) x 40 columns (each article)
# Each number = how similar your draft is to that article, 0 to 1

print("Raw similarity scores shape:", similarity_scores.shape)
print("----------------------------------------------------------------------------")

print(similarity_scores)

print("----------------------------------------------------------------------------")

# Flatten from (1, 40) shape into a plain list of 40 scores
scores = similarity_scores.flatten()
# .flatten() turns [[0.002, 0.0, ...]] into [0.002, 0.0, ...]
# — easier to work with when it's just one draft being compared

# Get the indices of the top 5 highest scores, sorted best-first
top_5_indices = np.argsort(scores)[::-1][:5]
# np.argsort(scores) -> returns indices that would sort scores LOW to HIGH
# [::-1] -> reverses it, so now HIGH to LOW
# [:5] -> keep only the first 5 (the best 5 matches)

print("\n--- Top 5 Recommended Articles ---\n")
for rank, idx in enumerate(top_5_indices, start=1):
    print("----------------------------------------------------------------------------")

    print(f"#{rank} | Score: {scores[idx]:.3f}")
    print("----------------------------------------------------------------------------")
    print(f"Title: {df_sample.loc[idx, 'title']}")
    print("----------------------------------------------------------------------------")
    print(f"Tags: {df_sample.loc[idx, 'tags']}")
    print("----------------------------------------------------------------------------")
    print("-" * 50)

# ============================================================
# DIAGNOSTIC: why did the #1 result score so high?
# ============================================================
# Find which words are contributing most to the Laravel article's high score
laravel_idx = top_5_indices[0]  # index of your #1 result
draft_words = vectorizer.get_feature_names_out()
draft_array = draft_vector.toarray()[0]
laravel_array = tfidf_matrix[laravel_idx].toarray()[0]

# Words present in BOTH the draft and the top-matched article
overlap = [(draft_words[i], draft_array[i], laravel_array[i])
           for i in range(len(draft_words))
           if draft_array[i] > 0 and laravel_array[i] > 0]

print("\nShared words driving this match:")
for word, draft_score, article_score in sorted(overlap, key=lambda x: -x[2]):
    print(f"  '{word}' — draft: {draft_score:.3f}, article: {article_score:.3f}")