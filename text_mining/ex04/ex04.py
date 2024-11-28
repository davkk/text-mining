# %%
import seaborn as sns
import tmtoolkit as tm
from matplotlib import gridspec
from matplotlib import pyplot as plt
from nltk.corpus import opinion_lexicon
from wordcloud import WordCloud

plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["axes.grid"] = True
plt.rc("figure", titlesize=22)

fig = plt.figure(figsize=(16, 16))
gs = gridspec.GridSpec(2, 2, figure=fig)

ax_emotion = fig.add_subplot(gs[:, 0])
ax_cloud = fig.add_subplot(gs[0, 1])
ax_pos = fig.add_subplot(gs[1, 1])


# %%
corp = tm.corpus.Corpus(
    {"Crime and Punishment": open("./crime-and-punishment.txt").read()},
    language="en",
)
tm.corpus.filter_clean_tokens(corp)
tm.corpus.to_lowercase(corp)


# %%
tokens = tm.corpus.tokens_table(corp)
top_3_nouns = (
    tokens[tokens["pos"] == "NOUN"]
    .value_counts()
    .reset_index()
    .head(3)["token"]
    .tolist()
)


# %%
corp_bi = tm.corpus.corpus_ngramify(corp, n=2, inplace=False)
tokens_bi = tm.corpus.tokens_table(corp_bi)

tokens_bi[["token", "noun"]] = tokens_bi["token"].str.split(" ", n=1, expand=True)

tokens_bi = tokens_bi[tokens_bi["noun"].isin(top_3_nouns)]

tokens_bi["pos"] = tokens_bi["pos"].apply(lambda x: x.split()[0])
tokens_bi["tag"] = tokens_bi["tag"].apply(lambda x: x.split()[0])

df = tokens_bi[["token", "noun", "pos"]]
df_uniq = df.drop_duplicates(subset=["token"])


# %%
pos_hist = df_uniq["pos"].value_counts().reset_index()
sns.barplot(
    x="pos",
    y="count",
    data=pos_hist,
    palette="viridis",
    ax=ax_pos,
)
ax_pos.set_title("POS histogram")


# %%
wordcloud = WordCloud(background_color="white").generate(" ".join(df_uniq["token"]))
ax_cloud.imshow(wordcloud, interpolation="bilinear")
ax_cloud.axis("off")
ax_cloud.set_title("Word Cloud", fontsize=16)


# %%
pos = set(opinion_lexicon.positive())
df["positive"] = df["token"].apply(lambda x: x.split()[0] in pos)

neg = set(opinion_lexicon.negative())
df["negative"] = df["token"].apply(lambda x: x.split()[0] in neg)

df = df[df["positive"] | df["negative"]]


# %%
emotions = (
    df.groupby(["token", "positive", "negative"]).size().reset_index(name="count")
)
emotions["score"] = (
    emotions["count"] * emotions["positive"] - emotions["count"] * emotions["negative"]
)
emotions = emotions.sort_values(by="score", ascending=False)

sns.barplot(
    x="score",
    y="token",
    data=emotions,
    orient="h",
    palette=["green" if score > 0 else "red" for score in emotions["score"]],
    ax=ax_emotion,
)
ax_emotion.grid(False, axis="x")
ax_emotion.grid(True, axis="y")
ax_emotion.set_title(f"Emotion score for words to left of top 3 nouns {top_3_nouns}")
ax_emotion.set_xlabel("emotion score (positive=green, negative=red)")


# %%
fig.suptitle("Results for Crime and Punishment by Fyodor Dostoyevsky")
fig.tight_layout()

fig.savefig("results.pdf")
