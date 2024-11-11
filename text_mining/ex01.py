# %%
import string
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import nltk
import numpy as np
import seaborn as sns
from nltk.corpus import inaugural
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer

nltk.download("inaugural")

num_speeches = 6
files = inaugural.fileids()[-num_speeches:]

speeches = [
    " ".join(
        [
            word.lower()
            for word in nltk.word_tokenize(inaugural.raw(file))
            if word.isalpha() and word not in string.punctuation
        ]
    )
    for file in files
]


# %%
def calc_matrices(*, vectors):
    n = len(speeches)

    dot_prods = np.zeros((n, n))
    cos_sims = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            vec1, vec2 = vectors[i], vectors[j]

            norm_prod = np.linalg.norm(vec1) * np.linalg.norm(vec2)
            norm_prod = norm_prod if norm_prod != 0 else 0

            dot_prods[i, j] = np.dot(vec1, vec2)
            cos_sims[i, j] = dot_prods[i, j] / norm_prod

    return dot_prods, cos_sims


vec_with = CountVectorizer()
dot_with, cos_with = calc_matrices(
    vectors=cast(
        csr_matrix,
        vec_with.fit_transform(speeches),
    ).toarray()
)

vec_without = CountVectorizer(stop_words="english")
dot_without, cos_without = calc_matrices(
    vectors=cast(
        csr_matrix,
        vec_without.fit_transform(speeches),
    ).toarray()
)


# %%
plots = [
    (dot_with, "Dot product with stopwords"),
    (dot_without, "Dot product without stopwords"),
    (cos_with, "Cosine similarity with stopwords"),
    (cos_without, "Cosine similarity without stopwords"),
]

fig, axs = plt.subplots(figsize=(8, 7), ncols=2, nrows=2)
axs = axs.flatten()

labels = [Path(file).stem.replace("-", " ") for file in files]

for ax, (matrix, title) in zip(axs, plots):
    sns.heatmap(
        matrix,
        ax=ax,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        xticklabels=labels,
        yticklabels=labels,
    )
    ax.set_title(title)

fig.tight_layout()
plt.show()
