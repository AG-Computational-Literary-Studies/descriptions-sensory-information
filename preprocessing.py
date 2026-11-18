# %% [markdown]
# # Preprocessing
#
# Pipeline:
# 1. Load the corpora and the BERT-annotated sentences.
# 2. Compute corpus statistics and the sentence-length filter.
# 3. Lemmatize the sentences (spaCy), dropping stopwords and non-alphabetic tokens.
# 4. Score each sentence with the Lancaster Sensorimotor Norms (four modalities)
#    and the Glasgow Imageability Norms, plus the lexical coverage of each norm.
# 5. Print summary statistics.
#
# Interactive script (percent format): run cell by cell in VS Code, Spyder,
# PyCharm or any editor that supports `# %%` cell markers.

# %%
import pandas as pd
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from tqdm import tqdm

# %% [markdown]
# ## Parameters

# %%
# Sentences with this many words or fewer are never treated as descriptions
MIN_SENTENCE_LENGTH = 5

# Keep sentences within +/- this many words of the median description length
LENGTH_WINDOW = 5

# %% [markdown]
# ## Load the Data

# %%
spatial_descriptions = pd.read_csv("spatial_descriptions.csv")
reference = pd.read_csv("reference.csv")

# Sentences annotated by the BERT classifier (run on a server)
gb_annotated = pd.read_csv("classifier_annotations_raw.csv")

# Remove potential duplicates
gb_annotated = gb_annotated.drop_duplicates(subset=["sentence_id"])

# Very short sentences (<= MIN_SENTENCE_LENGTH words) are labelled "not a description"
gb_annotated.loc[gb_annotated["length"] <= MIN_SENTENCE_LENGTH, "ANNOTATION"] = 0

# %% [markdown]
# ## Descriptive Statistics / Corpus Size

# %%
descriptions = gb_annotated[gb_annotated["ANNOTATION"] == 1]
positive_class_ratio = len(descriptions) / len(gb_annotated)

print("Complete corpus size:", len(gb_annotated))
print("Positive annotations:", len(descriptions))
print("Ratio of spatial descriptions:", positive_class_ratio)

# Share of spatial descriptions per text, averaged over texts
# (sentence_id format: "<prefix>-<text_id>-...")
gb_annotated["text_id"] = gb_annotated["sentence_id"].str.split("-").str[1]
ratio_per_text = (
    gb_annotated["ANNOTATION"].eq(1).groupby(gb_annotated["text_id"]).mean()
)
print("Mean ratio per text:", ratio_per_text.mean())

# %% [markdown]
# ## Filter to Relevant Sentences
#
# - unique sentences only
# - length within +/- `LENGTH_WINDOW` words of the median description length

gb_annotated = gb_annotated.drop_duplicates(subset=["sentence"], keep="first")


def filter_by_length(sentences, center, window=LENGTH_WINDOW):
    """Keep sentences whose length is within `center` +/- `window` words."""
    return sentences[sentences["length"].between(center - window, center + window)]


# Spatial descriptions (positive class)
spatial_descriptions_all = gb_annotated[gb_annotated["ANNOTATION"] == 1]
print("Identified Descriptions:", len(spatial_descriptions_all))

median_sentence_length = int(spatial_descriptions_all["length"].median())
print("Median Sentence Length:", median_sentence_length)

descriptions_filtered = filter_by_length(spatial_descriptions_all, median_sentence_length)
print("Relevant Descriptions:", len(descriptions_filtered))

# Reference sentences (negative class), filtered to the same length range
reference_all = gb_annotated[gb_annotated["ANNOTATION"] == 0]
reference_filtered = filter_by_length(reference_all, median_sentence_length)
print("Relevant Reference Sentences:", len(reference_filtered))


# %% [markdown]
# ## Lemmatize

# The parser and NER are not needed for lemmatization
nlp = spacy.load("en_core_web_lg", exclude=["parser", "ner"])
stopwords = {w.lower() for w in STOP_WORDS}


def lemmatize_sentences(sentences, stopwords=stopwords, nlp=nlp):
    """Add a `lemmatized_sentence` column (list of lemmas per sentence).

    Stopwords and tokens that are not purely alphabetic are removed.
    Sentences that fail to process get None.
    """
    sentences = sentences.copy()
    lemmatized = []

    for sentence in tqdm(sentences["sentence"].to_list()):
        try:
            lemmas = [token.lemma_ for token in nlp(sentence)]
            lemmatized.append(
                [lemma for lemma in lemmas if lemma.lower() not in stopwords and lemma.isalpha()]
            )
        except Exception as e:
            print(f"Error processing sentence: {sentence}. Error: {e}")
            lemmatized.append(None)

    sentences["lemmatized_sentence"] = lemmatized
    return sentences


spatial_descriptions = lemmatize_sentences(spatial_descriptions)
reference = lemmatize_sentences(reference)

# %% [markdown]
# ## Norm lookup helpers

def lookup_norm(word, norms):
    """Return the norm value for `word`, or None if the word is not covered.

    NOTE: the word only counts as covered if its form with trailing "s"
    characters removed is in `norms` (kept as in the original script). The
    exact form is preferred when both forms exist. This means words such as
    "glass" (exact form listed, stripped form "gla" not listed) are skipped.
    If that is not intended, use instead:
        key = word.lower()
        if key in norms: return norms[key]
        return norms.get(key.rstrip("s"))
    """
    key = word.lower()
    stem = key.rstrip("s")
    if stem in norms:
        return norms[key] if key in norms else norms[stem]
    return None


def lookup_scores(words, norms):
    """Norm values of all covered words in a list of lemmas."""
    scores = (lookup_norm(word, norms) for word in words)
    return [score for score in scores if score is not None]


def mean_or_none(values):
    return sum(values) / len(values) if values else None


# %% [markdown]
# ## Lancaster Sensorimotor Norms
#
# Source: https://link.springer.com/article/10.3758/s13428-019-01316-z

lc_df = pd.read_csv("lancaster_norms.csv")

# Output column -> column in the Lancaster file
LANCASTER_COLUMNS = {
    "auditory_mean": "Auditory.mean",
    "olfactory_mean": "Olfactory.mean",
    "visual_mean": "Visual.mean",
    "haptic_mean": "Haptic.mean",
}

# One {word: rating} dictionary per modality
lancaster_norms = {
    out_col: dict(zip(lc_df["Word"].str.lower(), lc_df[src_col]))
    for out_col, src_col in LANCASTER_COLUMNS.items()
}


def compute_lancaster_scores(sentences, norms=lancaster_norms):
    """Add the mean rating per modality and `coverage_lancaster` to each sentence.

    Coverage = share of lemmas found in the norms (based on the auditory
    dictionary; all modalities come from the same word list).
    """
    sentences = sentences.copy()
    means = {col: [] for col in norms}
    coverage = []

    for words in tqdm(sentences["lemmatized_sentence"]):
        words = words or []  # sentences that failed lemmatization (None)
        scores = {col: lookup_scores(words, norm) for col, norm in norms.items()}

        for col in norms:
            means[col].append(mean_or_none(scores[col]))
        coverage.append(len(scores["auditory_mean"]) / len(words) if words else 0)

    for col, values in means.items():
        sentences[col] = values
    sentences["coverage_lancaster"] = coverage
    return sentences


spatial_descriptions = compute_lancaster_scores(spatial_descriptions)
reference = compute_lancaster_scores(reference)

# %% [markdown]
# ## Glasgow Imagability Norms
#
# Source: https://link.springer.com/article/10.3758/s13428-018-1099-3

img_df = pd.read_csv("glasgow_norms.csv")
img_dict = dict(zip(img_df["Word"].str.lower(), img_df["IMAG_M"]))


def compute_glasgow_scores(sentences, norms=img_dict):
    """Add `imagability_mean` and `coverage_glasgow` to each sentence."""
    sentences = sentences.copy()
    imagability_means = []
    coverage = []

    for words in tqdm(sentences["lemmatized_sentence"]):
        words = words or []
        scores = lookup_scores(words, norms)
        imagability_means.append(mean_or_none(scores))
        coverage.append(len(scores) / len(words) if words else 0)

    sentences["imagability_mean"] = imagability_means
    sentences["coverage_glasgow"] = coverage
    return sentences


spatial_descriptions = compute_glasgow_scores(spatial_descriptions)
reference = compute_glasgow_scores(reference)

# %% [markdown]
# ## Coverage and Summary Statistics

# %%
corpora = {
    "spatial_descriptions": spatial_descriptions,
    "reference": reference,
}
corpora_with_both = {
    **corpora,
    "both": pd.concat([spatial_descriptions, reference], ignore_index=True),
}

# Lexical coverage of the norms
for name, data in corpora_with_both.items():
    print(f"Statistics for {name}:")
    for col in ["coverage_lancaster", "coverage_glasgow"]:
        print(f"  {col}: min={data[col].min()}, mean={data[col].mean()}, max={data[col].max()}")

# Rating distributions per corpus
for col in ["visual_mean", "auditory_mean", "olfactory_mean", "haptic_mean", "imagability_mean"]:
    print(f"Statistics for {col}:")
    for name, data in corpora.items():
        print(
            f"  {name}: min={data[col].min()}, mean={data[col].mean()}, "
            f"max={data[col].max()}, std={data[col].std()}"
        )

