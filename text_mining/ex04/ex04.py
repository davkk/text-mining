import tmtoolkit as tm
from matplotlib import pyplot as plt
from nltk.corpus import opinion_lexicon

plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["axes.grid"] = True
plt.rc("figure", titlesize=22)

corp = {
    "Letters From a Stoic": open("./letters-from-stoic.txt").read(),
}
corp = tm.corpus.Corpus(corp, language="en")
tm.corpus.filter_clean_tokens(corp)
tm.corpus.to_lowercase(corp)

tokens = tm.corpus.tokens_table(corp)
top_3_nouns = (
    tokens[tokens["pos"] == "NOUN"]
    .value_counts()
    .reset_index()
    .head(3)["token"]
    .tolist()
)
print(f"{top_3_nouns=}")

corp_bi = tm.corpus.corpus_ngramify(corp, n=2, inplace=False)
tokens_bi = tm.corpus.tokens_table(corp_bi)
tokens_bi = tokens_bi[tokens_bi["token"].apply(lambda x: x.split()[1] in top_3_nouns)]
tokens_bi = tokens_bi.drop_duplicates(subset=["token"])

pos = set(opinion_lexicon.positive())
tokens_bi["pos"] = tokens_bi["token"].apply(lambda x: x.split()[0] in pos)

neg = set(opinion_lexicon.negative())
tokens_bi["neg"] = tokens_bi["token"].apply(lambda x: x.split()[0] in neg)

df = tokens_bi[tokens_bi["pos"] | tokens_bi["neg"]]

print(df)
