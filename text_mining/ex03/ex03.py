# %%
import sys

import networkx as nx
import seaborn as sns
import tmtoolkit as tm
from matplotlib import pyplot as plt

plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["axes.grid"] = True
plt.rc("figure", titlesize=22)

with_stopwords = None
match sys.argv[1]:
    case "with":
        with_stopwords = True
    case "without":
        with_stopwords = False

assert with_stopwords is not None

# %%
corp = {
    "Letters From a Stoic": open("data/letters.txt").read(),
    "The Enchiridion": open("data/enchiridion.txt").read(),
}

corp = tm.corpus.Corpus(corp, language="en", load_features=[])

tm.corpus.filter_clean_tokens(corp, remove_stopwords=not with_stopwords)
tm.corpus.to_lowercase(corp)
tm.corpus.remove_tokens(corp, "")

corp_bi = tm.corpus.corpus_ngramify(corp, n=2, inplace=False)
tokens = tm.corpus.tokens_table(corp_bi)


# %%
fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(12, 11))

for [left, right], (doc, data) in zip(axs, tokens.groupby("doc")):
    data = data["token"].value_counts().reset_index().head(20)
    sns.barplot(
        data,
        y="token",
        x="count",
        orient="h",
        ax=left,
    )
    left.set_title(f"Histogram: {doc}")

    data = data["token"].str.split(expand=True).add_prefix("token")
    data = data[["token0", "token1"]].value_counts().reset_index().head(20)

    G = nx.from_pandas_edgelist(data, "token0", "token1", edge_attr="count")

    nx.draw_networkx(
        G,
        pos=nx.kamada_kawai_layout(G),
        node_color="plum",
        edge_color="lightgray",
        font_size=8,
        node_size=100,
        ax=right,
    )
    right.set_axis_off()
    right.set_title(f"Graph: {doc}")


# %%
fig.tight_layout(rect=(0, 0.03, 1, 0.95))

case = "with" if with_stopwords else "without"
fig.suptitle(f"Bigrams {case} stopwords")
plt.savefig(f"bigrams_{case}_stopwords.pdf")
