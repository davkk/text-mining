import os
from pathlib import Path

import matplotlib.pyplot as plt
import nltk
import spacy
import tmtoolkit as tm
from tmtoolkit.topicmod.tm_lda import compute_models_parallel
from wordcloud import WordCloud

nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
nltk.download("stopwords")

DATA = Path(__file__).parent / "data"


def read_text(file: str):
    with open(DATA / file) as text:
        return text.read()


def parse_word(word_freq: str) -> tuple[str, float]:
    [word, freq] = word_freq.split()
    return word, float(freq[1:-1])


def analyze(*, texts, title, axs, n_topics):
    print(f"creating corpus for {title}...")
    corpus = tm.corpus.Corpus(dict(texts), language="en")

    tm.corpus.filter_clean_tokens(corpus)
    tm.corpus.to_lowercase(corpus)
    tm.corpus.lemmatize(corpus)

    mat, _, vocab = tm.corpus.dtm(  # type: ignore
        corpus,
        return_doc_labels=True,
        return_vocab=True,
    )

    lda_params = dict(
        n_topics=n_topics,
        n_iter=1000,
    )

    print(f"computing models for {title}...")
    models = compute_models_parallel(mat, constant_parameters=lda_params)

    top_words = tm.topicmod.model_io.ldamodel_top_topic_words(
        models[0][1].topic_word_,
        vocab,
        10,
    ).T

    for idx, ax in enumerate(axs):
        t = dict(parse_word(w) for w in top_words[f"topic_{idx+1}"])
        wordcloud = WordCloud(background_color="white").generate_from_frequencies(t)
        ax.imshow(wordcloud, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(f"{title.capitalize()} - topic {idx + 1}")

    return mat


files = os.listdir(DATA)

dumas = [("dumas", read_text(file)) for file in files if file.startswith("dumas")]
austen = [("austen", read_text(file)) for file in files if file.startswith("austen")]

fig1, [axs_dumas, axs_austen, axs_both] = plt.subplots(nrows=3, ncols=2, figsize=(9, 6))
mat_dumas = analyze(texts=dumas, title="dumas", axs=axs_dumas, n_topics=2)
mat_austen = analyze(texts=austen, title="austen", axs=axs_austen, n_topics=2)
mat_both = analyze(texts=dumas + austen, title="combined", axs=axs_both, n_topics=2)
fig1.suptitle("Top 2 topics")
fig1.tight_layout()
fig1.savefig(Path(__file__).parent / "top2_topics.pdf")

fig2, axs_both = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
analyze(texts=dumas + austen, title="combined - top 3 topics", axs=axs_both, n_topics=3)
fig2.tight_layout()
fig2.savefig(Path(__file__).parent / "top3_topics_combined.pdf")


var_params = [dict(n_topics=k, alpha=1 / k) for k in range(2, 30)]
const_params = dict(n_iter=100)

models = tm.topicmod.tm_lda.compute_models_parallel(
    mat_both,
    varying_parameters=var_params,
    constant_parameters=const_params,
)

coh = [
    tm.topicmod.evaluate.metric_coherence_mimno_2011(
        models[i][1].topic_word_,
        mat_both,
        include_prob=True,
    ).mean()
    for i in range(len(models))
]
ntop = [models[i][1].n_topics for i in range(len(models))]

fig3, ax = plt.subplots()
ax.plot(ntop, coh, ".-")
ax.set_xlabel("Number of topics")
ax.set_ylabel("Coherence Score")
ax.set_title("Combined - coherence scores for different number of topics")
fig3.tight_layout()
fig3.savefig(Path(__file__).parent / "coherence_combined.pdf")
