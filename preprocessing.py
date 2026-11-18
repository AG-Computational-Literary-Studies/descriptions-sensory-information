# %% [markdown]
# # Preprocessing
# Interactive script (percent format): run cell by cell in VS Code, Spyder,
# PyCharm or any editor that supports `# %%` cell markers.

# %%
import os
import spacy

import pandas as pd

from tqdm import tqdm

# %% [markdown]
# ### Preprocess sentences annotated by Bert classifier on server

# %%
gb_annotated = pd.read_csv("classifier_annotations_raw.csv")

# remove potential duplicates
gb_annotated = gb_annotated.drop_duplicates(subset=["sentence_id"])

# map BERT output to actual classes
gb_annotated['ANNOTATION'] = gb_annotated['ANNOTATION'].map({'[0]': 0, '[1]': 1})

# filter to sentences that are at least 5 words long
gb_annotated.loc[gb_annotated['length'] <= 5, 'ANNOTATION'] = 0

# %% [markdown]
# ### Count

# %%
descriptions = gb_annotated[gb_annotated['ANNOTATION'] == 1]
positive_class_ratio = len(descriptions) / len(gb_annotated)

print(len(gb_annotated))  # complete corpus size
print(len(descriptions))  # all positive annotations
print(positive_class_ratio)  # ratio of spatial descriptions

gb_annotated["text_id"] = gb_annotated["sentence_id"].apply(lambda x: x.split("-")[1])
ratio_per_text = gb_annotated["ANNOTATION"].eq(1).groupby(gb_annotated["text_id"]).mean()
mean_text_ratio = ratio_per_text.mean()
print(mean_text_ratio)

# %% [markdown]
# ### Filter to Relevant Sentences
#
# - unique
# - +/- 5 words from mean

# %%
gb_annotated = gb_annotated.drop_duplicates(subset=['sentence'], keep='first')

len(gb_annotated[gb_annotated['ANNOTATION'] == 1])

# %%
spatial_descriptions_all = gb_annotated[gb_annotated['ANNOTATION'] == 1]
median_sentence_length = int(spatial_descriptions_all['length'].median())
print(median_sentence_length)


descriptions_filtered = spatial_descriptions_all[
    (spatial_descriptions_all['length'] >= median_sentence_length - 5) &
    (spatial_descriptions_all['length'] <= median_sentence_length + 5)
]
print(len(descriptions_filtered))


# filter reference
reference_all = gb_annotated[gb_annotated['ANNOTATION'] == 0]
reference_filtered = reference_all[
    (reference_all['length'] >= median_sentence_length - 5) &
    (reference_all['length'] <= median_sentence_length + 5)
]
print(len(reference_filtered))

# %% [markdown]
# The filtered descriptions still contain some false positives (person
# descriptions) which were further filtered out using an LLM.

# %% [markdown]
# ### How many sentences per text do we finally consider in the analyses?

# %%
spatial_descriptions = pd.read_csv("/Users/esitter/Documents/repos/descriptions-sensory-imageability/spatial_descriptions.csv")
reference = pd.read_csv("/Users/esitter/Documents/repos/descriptions-sensory-imageability/reference.csv")

gb_annotated = pd.read_csv("/Users/esitter/Documents/repos/descriptions-sensory-imageability/classifier_annotations_raw.csv")
gb_annotated = gb_annotated.drop_duplicates(subset=["sentence_id"])

# %%
gb_annotated["Book_ID"] = gb_annotated["sentence_id"].apply(lambda x: x.split("-")[1])
keys = ["Book_ID", "sentence_id"]

# %%
# filtered sentences ratio (general) per text

all_considered = pd.concat([spatial_descriptions, reference], ignore_index=True)

for df in (gb_annotated, all_considered):
    for k in keys:
        df[k] = df[k].astype(str).str.strip()

ga = gb_annotated[keys].drop_duplicates()
sd = all_considered[keys].drop_duplicates()

merged = ga.merge(sd, on=keys, how="left", indicator=True)
merged["in_both"] = merged["_merge"].eq("both")

ratio_per_book = merged.groupby("Book_ID")["in_both"].mean()
mean_ratio = ratio_per_book.mean()

print(mean_ratio)

# %% [markdown]
# ### Lemmatize

# %%
nlp = spacy.load("en_core_web_lg", exclude=["parser", "ner"])
stopwords = spacy.lang.en.stop_words.STOP_WORDS
stopwords = [w.lower() for w in stopwords]

# %%
def lemmatize_sentences(filtered_sentences, stopwords=stopwords, nlp=nlp):

    sentences_lemmatized = []

    for sentence in tqdm(filtered_sentences['sentence'].to_list()):
        try:
            doc = nlp(sentence)
            lemmatized_sentence = [token.lemma_ for token in doc]
            lemmatized_sentence = [token for token in lemmatized_sentence if token.lower() not in stopwords and token.isalpha()]
            sentences_lemmatized.append(lemmatized_sentence)
        except Exception as e:
            print(f"Error processing sentence: {sentence}. Error: {e}")
            sentences_lemmatized.append(None)

    filtered_sentences['lemmatized_sentence'] = sentences_lemmatized

    return filtered_sentences


spatial_descriptions = lemmatize_sentences(spatial_descriptions)
reference = lemmatize_sentences(reference)

# %% [markdown]
# ### Lancaster Sensorimotor Norms

# %%
lc_df = pd.read_csv("lancaster_norms.csv")  # https://link.springer.com/article/10.3758/s13428-019-01316-z
auditory_dict = dict(zip(lc_df['Word'].str.lower(), lc_df['Auditory.mean']))
olfactory_dict = dict(zip(lc_df['Word'].str.lower(), lc_df['Olfactory.mean']))
visual_dict = dict(zip(lc_df['Word'].str.lower(), lc_df['Visual.mean']))
haptic_dict = dict(zip(lc_df['Word'].str.lower(), lc_df['Haptic.mean']))

# %%
def compute_lancaster_scores(filtered_sentences, auditory_dict=auditory_dict, olfactory_dict=olfactory_dict, visual_dict=visual_dict, haptic_dict=haptic_dict):

    auditory_means = []
    olfactory_means = []
    visual_means = []
    haptic_means = []
    coverage_lancaster = []

    for words in tqdm(filtered_sentences['lemmatized_sentence']):
        auditory_scores = [auditory_dict[word.lower()] if word.lower() in auditory_dict else auditory_dict[word.lower().rstrip("s")] for word in words if word.lower().rstrip("s") in auditory_dict]
        olfactory_scores = [olfactory_dict[word.lower()] if word.lower() in olfactory_dict else olfactory_dict[word.lower().rstrip("s")] for word in words if word.lower().rstrip("s") in olfactory_dict]
        visual_scores = [visual_dict[word.lower()] if word.lower() in visual_dict else visual_dict[word.lower().rstrip("s")] for word in words if word.lower().rstrip("s") in visual_dict]
        haptic_scores = [haptic_dict[word.lower()] if word.lower() in haptic_dict else haptic_dict[word.lower().rstrip("s")] for word in words if word.lower().rstrip("s") in haptic_dict]

        auditory_means.append(sum(auditory_scores) / len(auditory_scores) if auditory_scores else None)
        olfactory_means.append(sum(olfactory_scores) / len(olfactory_scores) if olfactory_scores else None)
        visual_means.append(sum(visual_scores) / len(visual_scores) if visual_scores else None)
        haptic_means.append(sum(haptic_scores) / len(haptic_scores) if haptic_scores else None)

        coverage_lancaster.append(len(auditory_scores) / len(words) if words else 0)

    filtered_sentences['visual_mean'] = visual_means
    filtered_sentences['auditory_mean'] = auditory_means
    filtered_sentences['olfactory_mean'] = olfactory_means
    filtered_sentences['haptic_mean'] = haptic_means
    filtered_sentences["coverage_lancaster"] = coverage_lancaster

    return filtered_sentences


spatial_descriptions = compute_lancaster_scores(spatial_descriptions)
reference = compute_lancaster_scores(reference)

# %% [markdown]
# ### Glasgow Imagability Norms

# %%
img_df = pd.read_csv("glasgow_norms.csv")  # https://link.springer.com/article/10.3758/s13428-018-1099-3
img_dict = dict(zip(img_df['Word'].str.lower(), img_df['IMAG_M']))

# %%
def compute_glasgow_scores(filtered_sentences, img_dict=img_dict):

    imagability_means = []
    coverage_glasgow = []

    for words in tqdm(filtered_sentences['lemmatized_sentence']):
        img_scores = [img_dict[word.lower()] if word.lower() in img_dict else img_dict[word.lower().rstrip("s")] for word in words if word.lower().rstrip("s") in img_dict]
        if img_scores:
            imagability_means.append(sum(img_scores) / len(img_scores))
            coverage_glasgow.append(len(img_scores) / len(words))
        else:
            imagability_means.append(None)
            coverage_glasgow.append(0)

    filtered_sentences['imagability_mean'] = imagability_means
    filtered_sentences["coverage_glasgow"] = coverage_glasgow

    return filtered_sentences


spatial_descriptions = compute_glasgow_scores(spatial_descriptions)
reference = compute_glasgow_scores(reference)

# %% [markdown]
# ### Descriptive Statistics

# %%
# Coverage

for df_name, df in [("spatial_descriptions", spatial_descriptions),
                    ("reference", reference),
                    ("both", pd.concat([spatial_descriptions, reference], ignore_index=True))]:
    print(f"Statistics for {df_name}:")
    for col in ["coverage_lancaster", "coverage_glasgow"]:
        print(f"  {col}: min={df[col].min()}, mean={df[col].mean()}, max={df[col].max()}")

# %%
for col in ["visual_mean", "auditory_mean", "olfactory_mean", "haptic_mean", "imagability_mean"]:
    print(f"Statistics for {col}:")
    for df in [spatial_descriptions, reference]:
        df_name = "spatial_descriptions" if df is spatial_descriptions else "reference"
        min = df[col].min()
        mean = df[col].mean()
        max = df[col].max()
        sd = df[col].std()
        print(f"  {df_name}: min={min}, mean={mean}, max={max}, std={sd}")
