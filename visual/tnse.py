from sklearn import datasets
from openTSNE import TSNE



import matplotlib.pyplot as plt

from visual import tsneutil
# import tsneutil

iris = datasets.load_iris()
x, y = iris["data"], iris["target"]

def draw(x,y):
    tsne = TSNE(
        perplexity=20,
        n_iter=500,
        metric="euclidean",
        # callbacks=ErrorLogger(),
        n_jobs=8,
        random_state=42,
    )
    embedding = tsne.fit(x)
    tsneutil.plot(embedding, y, colors=tsneutil.MOUSE_10X_COLORS)

    print('end')
# draw(x,y)