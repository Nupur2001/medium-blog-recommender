import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
# pandas: for loading/filtering your CSV data
# TfidfVectorizer: the tool that turns text into numeric vectors
#
# WHAT IS TfidfVectorizer, IN SIMPLE WORDS?
# Think of it as a machine with one job: read a bunch of text, and hand you
# back a spreadsheet of numbers instead. Computers can't do math on words —
# only on numbers — so this is the translator between "text" and "numbers
# a computer can actually compare."
#
# Example with 3 tiny articles:
#   A: "cats are great pets"
#   B: "dogs are great pets"
#   C: "python is a programming language"
#
# TfidfVectorizer does two things:
#   1. Builds a vocabulary list of every unique word across all articles:
#      cats, are, great, pets, dogs, python, is, a, programming, language
#   2. Scores each article against that vocabulary — for each article,
#      it fills in a number per word: "how important is this word to THIS article?"
#
# Article A might become something like:
#   cats: 0.7   are: 0.1   great: 0.3   pets: 0.3   dogs: 0.0   python: 0.0 ...
#
# Notice "cats" scores high (distinctive to A), "are" scores low (common,
# boring word), and "dogs"/"python" score ZERO (don't appear in A at all).
# Every article becomes a row of numbers like this — called a "vector" —
# the article's numeric fingerprint.
#
# WHY "TF-IDF" SPECIFICALLY?
#   TF  (Term Frequency):          how many times does this word show up
#                                   in THIS article?
#   IDF (Inverse Document Freq.):  how RARE is this word across ALL articles?
# A word only gets a high score if it's BOTH frequent in this one article
# AND rare elsewhere — that's what makes it good at finding words that
# actually describe what an article is uniquely about, not just common
# words everyone uses.

df = pd.read_csv("medium_articles.csv")
# Loads the FULL 192k-row dataset into memory as a DataFrame

tech_keywords = ["Technology", "Artificial Intelligence", "Machine Learning",
                  "Data Science", "Programming", "Tech", "Software", "AI"]
# A plain Python list of words we consider "tech-related" —
# these are what we'll search for INSIDE the tags column

mask = df["tags"].str.contains("|".join(tech_keywords), case=False, na=False)
# .str.contains(...) checks EVERY row's tags string for a match
# "|".join(tech_keywords) turns the list into "Technology|Artificial Intelligence|...|AI"
#   — the | means "OR" in pattern matching, so this matches ANY of these words
# case=False -> ignore uppercase/lowercase differences
# na=False -> if tags is missing/blank, treat it as "no match" instead of crashing
# Result: `mask` is a column of True/False, one per row — True if that row is tech-related

df_sample = df[mask].sample(40, random_state=42).reset_index(drop=True)
# df[mask] -> keeps ONLY the rows where mask is True (the filtering step)
# .sample(40, random_state=42) -> randomly picks 40 rows from that filtered set
#   random_state=42 makes the "random" pick reproducible — same 40 rows every run
# .reset_index(drop=True) -> renumbers rows cleanly as 0,1,2,3...
#   (instead of keeping scattered original index numbers like 51649, 68568)

df_sample["combined_text"] = df_sample["title"].fillna("") + " " + df_sample["text"].fillna("")
# Creates a NEW column by joining title + text together into one string per article
# .fillna("") -> replaces any missing/NaN values with an empty string first,
#   because you can't concatenate text with NaN (it would crash or produce "nan" as text)
# Why combine title+text? Because both carry meaning — the title is often a strong
#   signal on its own, and the body gives context TF-IDF can use

vectorizer = TfidfVectorizer(stop_words="english", max_features=500)
# Creates the TF-IDF "machine", configured with two settings:
# stop_words="english" -> automatically ignores common filler words
#   (the, is, and, a, to...) since they carry no topical meaning
# max_features=500 -> only keep the top 500 most informative words across
#   the whole sample — keeps the matrix small and manageable while learning
#   (otherwise it would track EVERY unique word, which could be thousands)

tfidf_matrix = vectorizer.fit_transform(df_sample["combined_text"])
# THIS is where the actual math happens. Two things occur in one call:
# .fit(...)       -> reads all 40 articles, learns the vocabulary (which 500 words matter)
# .transform(...) -> converts each article into its TF-IDF numeric vector
# fit_transform does both in one step since we're doing it on the same data
#
# Result: tfidf_matrix is a spreadsheet of numbers — one row per article,
# one column per tracked word. Each article is now a numeric "fingerprint"
# instead of raw text, ready to be compared to other fingerprints (Step 3).

print("Matrix shape:", tfidf_matrix.shape)
# Confirms: (40, 500) -> 40 articles, each described by 500 numbers

print("Sample vocabulary:", vectorizer.get_feature_names_out()[:20])
# Shows you the actual WORDS behind those 500 numbers (alphabetically, first 20)
# This is just a sanity check — "did TF-IDF learn sensible, topic-relevant words?"

# ONE-SENTENCE SUMMARY:
# TfidfVectorizer turns "this article is about cats" into "here's a list of
# 500 numbers that mathematically represent what this article is about" —
# so a computer can later compare two articles by comparing their
# number-lists instead of trying to "read" them.