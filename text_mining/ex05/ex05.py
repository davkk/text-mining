import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tmtoolkit as tm
from sklearn import metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

bbc = pd.read_csv(
    "https://jsienkiewicz.pl/TEXT/lab/data_bbc.csv",
    on_bad_lines="skip",
    encoding="ISO-8859-1",
)

pos_comments = bbc[bbc["emo"] == 1]
neg_comments = bbc[bbc["emo"] == -1]
obj_comments = bbc[bbc["emo"] == 0]

bbc = pd.concat(
    [
        pos_comments.sample(500, random_state=420),
        neg_comments.sample(500, random_state=420),
        obj_comments.sample(1000, random_state=420),
    ]
)

bbc["emo"] = bbc["emo"].replace({-1: 1})  # type: ignore
print(bbc["emo"].value_counts())  # type: ignore

bbc["text"] = bbc["text"].fillna("").astype(str)  # type: ignore

corp = tm.corpus.Corpus(
    dict(zip(map(str, range(len(bbc))), bbc.text)),
    language="en",
    load_features=[],
)
tm.corpus.set_document_attr(
    corp,
    attrname="emo",
    data=dict(zip(corp.doc_labels, bbc.emo)),
)

tm.corpus.filter_clean_tokens(corp, remove_numbers=True)
tm.corpus.to_lowercase(corp)

tm.corpus.print_summary(corp)


def train_tfidf():
    vectorizer = TfidfVectorizer(min_df=3, lowercase=True, stop_words="english")
    X = vectorizer.fit_transform(bbc.text)

    x_train, x_test, y_train, y_test = train_test_split(
        X,
        bbc.emo,
        test_size=0.3,
        stratify=bbc.emo,
        random_state=420,
    )

    svm_clf = svm.SVC(kernel="linear", C=100, class_weight="balanced")
    svm_clf.fit(x_train, y_train)

    print(metrics.classification_report(y_test, svm_clf.predict(x_test)))
    return metrics.confusion_matrix(y_test, svm_clf.predict(x_test))


def train_dtm():
    mat, doc_labels, vocab = tm.corpus.dtm(  # type: ignore
        corp, return_doc_labels=True, return_vocab=True
    )
    z = tm.bow.dtm.dtm_to_dataframe(mat, doc_labels, vocab).T

    ind_words = z.sum(axis=1) >= 3
    mat = mat[:, ind_words]
    z = tm.bow.dtm.dtm_to_dataframe(
        mat,
        doc_labels,
        np.array(vocab)[ind_words],
    ).T

    ind_docs = z.sum(axis=0) >= 3
    mat = mat[ind_docs, :]  # type: ignore
    z = tm.bow.dtm.dtm_to_dataframe(
        mat,
        np.array(doc_labels)[ind_docs],
        np.array(vocab)[ind_words],
    )

    emo_class = np.array(bbc.emo)[ind_docs]
    x_train, x_test, y_train, y_test = train_test_split(
        z,
        emo_class,
        stratify=emo_class,
        test_size=0.3,
    )

    svm_clf = svm.SVC(kernel="linear", C=100, class_weight="balanced")
    svm_clf.fit(x_train, y_train)

    print(metrics.classification_report(y_test, svm_clf.predict(x_test)))
    return metrics.confusion_matrix(y_test, svm_clf.predict(x_test))


fig, axs = plt.subplots(ncols=2, nrows=1, figsize=(12, 6))
[dtm, tfidf] = axs

metrics.ConfusionMatrixDisplay(train_dtm()).plot(ax=dtm)
metrics.ConfusionMatrixDisplay(train_tfidf()).plot(ax=tfidf)

fig.suptitle("Subjective (1 & -1) vs Objective (0) Classification")
dtm.set_title("DTM")
tfidf.set_title("TF-IDF")

plt.savefig("conf-mat.pdf")
plt.show()
