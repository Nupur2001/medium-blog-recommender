# Learnings

This file is where I log what I find while building this project — bugs, surprising results, limitations of the methods I'm using, and what I learned from digging into them. The goal isn't to hide the rough edges, it's to show the actual process: what I tried, what went wrong, why it went wrong, and what I'd do differently.

---

## TF-IDF "model" ambiguity (Step 3 — similarity testing)

**What happened:** When I tested my recommender with a draft about AI/LLMs, it recommended a Laravel (PHP) article as the #1 match.

**Why it happened:** I checked why, and it turned out both texts used the word "model" — but in my draft it meant "AI model" and in the Laravel article it meant "database model" (Laravel's Eloquent models). My method (TF-IDF) just matches words, it doesn't understand meaning, so it got confused by this one word.

**Takeaway:** This is a known limitation of TF-IDF — it matches on literal word overlap, not on meaning or context, so a single ambiguous word (a word with two unrelated meanings) can produce a confident-looking but wrong result. A future improvement would be switching to semantic embeddings (e.g. sentence-transformers), which represent meaning rather than just literal words, and would likely not confuse "AI model" with "database model."
