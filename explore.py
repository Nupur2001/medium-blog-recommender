import pandas as pd

df = pd.read_csv("medium_articles.csv")

# Basic shape and structure
print("DataFrame shape:", df.shape)
print("----------------------------------------------------------------------------")
print("DataFrame columns:", df.columns.to_list())
print("----------------------------------------------------------------------------")
print("DataFrame info:",df.info())
print("----------------------------------------------------------------------------")
print("First 3 rows:",df.head(3))
print("----------------------------------------------------------------------------")

# Filter to rows where tags mention tech/AI-related topics
tech_keywords = ["Technology", "Artificial Intelligence", "Machine Learning","Data Science", "Programming", "Tech", "Software", "AI"]

mask = df["tags"].str.contains("|".join(tech_keywords), case=False, na=False)
df_tech = df[mask]

print("Rows matching tech topics:", df_tech.shape[0])

# Now sample 40 from this filtered, topic-relevant subset
df_sample = df_tech.sample(min(40, len(df_tech)), random_state=15)
print(df_sample[['title', 'tags']])
print("----------------------------------------------------------------------------")



# Look at one full row to understand what you're working with
# print("First row:",df.iloc[0])