import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def pca_tsne_plots(X, y):
    """
    Visualizes multi-dimensional feature space in 2D using PCA and t-SNE projections.
    Helps inspect class separability between normal and fraudulent transactions.
    """
    pca = PCA(n_components=2, random_state=42)
    pca_sample = X.sample(min(50000, len(X)), random_state=42)
    y_pca_sample = y.loc[pca_sample.index]
    X_pca = pca.fit_transform(pca_sample)

    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[y_pca_sample == 0, 0], X_pca[y_pca_sample == 0, 1],
                s=5, alpha=0.5)
    plt.scatter(X_pca[y_pca_sample == 1, 0], X_pca[y_pca_sample == 1, 1],
                s=10, alpha=0.8, c="red")
    plt.title("PCA Projection (Large Sample)")
    plt.show()

    tsne_sample = X.sample(min(10000, len(X)), random_state=42)
    y_tsne_sample = y.loc[tsne_sample.index]
    X_tsne = TSNE(n_components=2, random_state=42).fit_transform(tsne_sample)

    plt.figure(figsize=(8, 6))
    plt.scatter(X_tsne[y_tsne_sample == 0, 0], X_tsne[y_tsne_sample == 0, 1],
                s=5, alpha=0.5)
    plt.scatter(X_tsne[y_tsne_sample == 1, 0], X_tsne[y_tsne_sample == 1, 1],
                s=10, alpha=0.8, c="red")
    plt.title("t-SNE Projection (Large Sample)")
    plt.show()