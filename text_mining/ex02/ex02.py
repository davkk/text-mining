# %%
import sys

import matplotlib.gridspec as gridspec
import numpy as np
import scipy
import seaborn as sns
import statsmodels.formula.api as sm
import tmtoolkit as tm
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.ticker import LogLocator

SHUFFLE = len(sys.argv) > 1


# %%
fig = plt.figure(figsize=(16, 8))
gs = gridspec.GridSpec(2, 4, figure=fig)

axs_top = []
for tile in range(2):
    ax = fig.add_subplot(gs[tile, 0])
    ax.set_aspect("equal")
    axs_top.append(ax)

axs_bot = []
for tile in range(2):
    ax = fig.add_subplot(gs[tile, 1])
    ax.set_aspect("equal")
    axs_bot.append(ax)

ax_corr = fig.add_subplot(gs[:, 2:])
ax_corr.set_aspect("equal")


# %%
def get_heaps(*, path: str, lang: str, axs: list[Axes]):
    # tworzenie korpusu dokumentow
    corp = tm.corpus.Corpus.from_folder(
        path,
        language=lang,
        load_features=[],
    )

    # czyszczenie tesktu
    tm.corpus.remove_punctuation(corp)
    tm.corpus.to_lowercase(corp)
    tm.corpus.remove_tokens(corp, "")

    heaps = tm.corpus.tokens_table(corp)

    for idx, (doc, data) in enumerate(heaps.groupby("doc")):
        if SHUFFLE:
            # mieszanie kolejnosci slow
            data["position"] = data["position"].sample(frac=1).values
            data.sort_values(by="position", inplace=True)

        data["position"] += 1

        data["V"] = ~data.duplicated(subset=["doc", "token"])
        data["V"] = data.groupby("doc")["V"].cumsum()

        # fitowanie do otrzymanych punktow z prawa Heapsa
        model = sm.ols("np.log10(V)~np.log10(position)", data=data).fit()
        B, beta = model.params

        if idx == 0 or idx == 3:
            x = np.logspace(np.log10(1), np.log10(data.max()["position"]), 100)
            y = 10**B * x**beta

            sns.scatterplot(
                data,
                x="position",
                y="V",
                marker=".",
                edgecolor="none",
                ax=axs[idx % 2],
            )

            axs[idx % 2].plot(x, y, "--", color="gray", alpha=0.5)

            axs[idx % 2].set_title(f"{lang}, book {doc}")
            axs[idx % 2].set_xscale("log")
            axs[idx % 2].set_yscale("log")
            axs[idx % 2].xaxis.set_major_locator(LogLocator(base=10.0, numticks=3))
            axs[idx % 2].yaxis.set_major_locator(LogLocator(base=10.0, numticks=3))

        yield beta


beta_de = list(get_heaps(path="data/de", lang="de", axs=axs_top))
beta_fr = list(get_heaps(path="data/fr", lang="fr", axs=axs_bot))

# %%
# liczenie korelacji
r, p = scipy.stats.pearsonr(beta_de, beta_fr)
print(f"{r=}, {p=}")

ax_corr.plot(
    [0.6, 0.8],
    [0.6, 0.8],
    "--",
    color="gray",
    alpha=0.5,
    label="$y = x$",
)

sns.scatterplot(
    x=beta_de,
    y=beta_fr,
    label=f"r = {r:.5f}, p = {p:.5f}",
    ax=ax_corr,
)

if SHUFFLE:
    ax_corr.set_title(
        "Heaps' Law Coefficient: German (de) vs. French (fr), shuffled word positions"
    )
else:
    ax_corr.set_title("Heaps' Law Coefficient: German (de) vs. French (fr)")

ax_corr.set_xlabel(r"$\beta$ (de)")
ax_corr.set_ylabel(r"$\beta$ (fr)")

plt.tight_layout()

if SHUFFLE:
    plt.savefig("result_shuffled.pdf")
else:
    plt.savefig("result.pdf")
