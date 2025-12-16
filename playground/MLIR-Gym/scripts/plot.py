import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

np.random.seed(42)

date = datetime.datetime.now().strftime("%Y-%m-%d")


def savefig(title: str, out_dir: str = "./snapshots", suffix: str = ""):
    """Save the current matplotlib figure to a file."""
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if suffix:
        filename = f"{title.replace(' ', '_')}_{suffix}_{timestamp}.png"
    else:
        filename = f"{title.replace(' ', '_')}_{timestamp}.png"
    path = os.path.join(out_dir, filename)
    plt.tight_layout()
    plt.savefig(path)
    print(f"[Saved] {path}")
    plt.close()


def extract_file_name(file_path: str):
    file_ext_path = file_path.split("/")[-1]
    file_name = file_ext_path.split(".")[0]
    return file_name


def plot_metrics(
    file, step, ob_mode, time_taken, episode_rewards, episode_lengths, window=100
):
    plt.figure(figsize=(12, 8))

    # Convert deques to lists for plotting
    episodes = np.arange(len(episode_rewards))
    time_taken = list(time_taken)
    episode_rewards = list(episode_rewards)
    episode_lengths = list(episode_lengths)

    # Calculate moving averages
    def moving_average(data, window):
        return [np.mean(data[max(0, i - window) : i + 1]) for i in range(len(data))]

    # Create subplots
    ax1 = plt.subplot(3, 1, 1)
    ax2 = plt.subplot(3, 1, 2)
    ax3 = plt.subplot(3, 1, 3)

    # Plot Episode Rewards
    ax1.plot(episodes, episode_rewards, label="Raw", alpha=0.3)
    ax1.plot(
        episodes,
        moving_average(episode_rewards, window),
        label=f"{window}-episode MA",
        color="orange",
    )
    ax1.set_ylabel("Total Reward")
    ax1.set_title("Training Progress")
    ax1.grid(True)
    ax1.legend()

    # Plot Time Taken
    ax2.plot(episodes, time_taken)
    ax2.set_ylabel("Time (seconds)")
    ax2.set_title("Time per Episode")
    ax2.grid(True)

    # Plot Episode Lengths
    ax3.plot(episodes, episode_lengths, label="Raw", alpha=0.3)
    ax3.plot(
        episodes,
        moving_average(episode_lengths, window),
        label=f"{window}-episode MA",
        color="green",
    )
    ax3.set_xlabel("Episodes")
    ax3.set_ylabel("Steps")
    ax3.set_title("Episode Lengths")
    ax3.grid(True)
    ax3.legend()

    plt.tight_layout()
    plt.savefig(
        f"snapshots/{extract_file_name(file)}_{ob_mode}_training_metrics_{len(episodes)}_{step}_{date}.png"
    )
    plt.show()


def plot_tsne(embeddings, metadata, suffix=""):
    """
    Plot a t-SNE visualization of embeddings.
    metadata is expected to be a list of tuples, where the first element (e.g. episode or state label)
    will be used for color-coding.
    """
    embeddings = np.array(embeddings)
    n_samples = embeddings.shape[0]
    if n_samples < 2:
        print(f"Warning: Not enough samples ({n_samples}) for t-SNE. Skipping plot.")
        return

    # Select perplexity based on available samples
    perplexity = min(30, max(1, n_samples - 1))
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    X_tsne = tsne.fit_transform(embeddings[:, 0, :].astype(np.float64))
    try:
        labels = metadata
    except Exception:
        labels = np.zeros(n_samples)

    plt.figure()
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, cmap="viridis")
    plt.colorbar(scatter, label="Label (e.g., episode or state)")
    plt.title("t-SNE of IR Embeddings")
    savefig("tsne_visualization", suffix=suffix)
    plt.show()
    plt.close()


def plot_pca(embeddings, suffix=""):
    """
    Plot the first two PCA components and plot the per-dimension variance.
    """
    embeddings = np.array(embeddings)
    embeddings = embeddings[:, 0, :] if len(embeddings.shape) == 3 else embeddings
    embeddings = np.array(embeddings, dtype=np.float64)
    n_samples, n_features = embeddings.shape
    max_components = min(n_samples, n_features)

    if max_components < 2:
        print(
            f"Warning: Not enough samples/features ({n_samples},{n_features}) for PCA"
        )
        return

    n_components = min(10, max_components - 1)
    pca = PCA(n_components=n_components)
    try:
        X_pca = pca.fit_transform(embeddings)
        print("PCA explained variance ratio:", pca.explained_variance_ratio_)
        print(
            "Cumulative explained variance:", np.cumsum(pca.explained_variance_ratio_)
        )

        # Plot the first two PCA components
        plt.figure()
        plt.scatter(X_pca[:, 0], X_pca[:, 1])
        plt.title("First 2 PCA Components")
        savefig("pca_2d_visualization", suffix=suffix)
        plt.show()
        plt.close()

        # Plot the per-dimension variance (in descending order)
        dim_variances = np.var(embeddings, axis=0)
        sorted_variances = np.sort(dim_variances)[::-1]
        plt.figure()
        plt.plot(sorted_variances, marker="o")
        plt.xlabel("Dimension (sorted)")
        plt.ylabel("Variance")
        plt.title("Descending Variance per Dimension")
        savefig("variance_per_dim", suffix=suffix)
        plt.show()
        plt.close()
    except Exception as e:
        print(f"PCA failed: {str(e)}")
        return


def plot_similarity_heatmap(embeddings, num_samples_to_show=50, suffix=""):
    """
    Plot a heatmap of pairwise cosine similarity among a random subset of embeddings.
    """
    embeddings = np.array(embeddings)
    embeddings = embeddings[:, 0, :] if len(embeddings.shape) == 3 else embeddings
    embeddings = np.array(embeddings, dtype=np.float64)
    N = embeddings.shape[0]
    idxs = np.random.choice(N, size=min(N, num_samples_to_show), replace=False)
    subset = embeddings[idxs]

    # Normalize the embeddings
    norms = np.linalg.norm(subset, axis=1, keepdims=True)
    normed_subset = subset / (norms + 1e-9)
    similarity = np.dot(normed_subset, normed_subset.T)

    plt.figure()
    plt.imshow(similarity, aspect="auto", cmap="viridis")
    plt.colorbar(label="Cosine Similarity")
    plt.title("Cosine Similarity Heatmap")
    savefig("cosine_similarity_heatmap", suffix=suffix)
    plt.show()
    plt.close()


def analyze_embeddings_with_classifier(embeddings, labels):
    """
    Train a logistic regression classifier on the embeddings to predict the given labels.
    This provides a quantitative measure of how distinguishable the embeddings are.
    Assumes embeddings is a numpy array of shape (n_samples, n_features) and labels is an array-like of shape (n_samples,).
    """
    from sklearn.model_selection import train_test_split

    embeddings = np.array(embeddings)
    embeddings = embeddings[:, 0, :] if len(embeddings.shape) == 3 else embeddings
    embeddings = np.array(embeddings, dtype=np.float64)
    labels = np.array(labels)

    # Split the data for evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, labels, test_size=0.3, random_state=42
    )

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Logistic Regression classifier accuracy on embeddings: {acc:.4f}")


def plot_gradient_norms(norms, suffix=""):
    plt.figure(figsize=(10, 5))
    plt.plot(norms)
    plt.xlabel("Episodes")
    plt.ylabel("Gradient Norm")
    plt.title("Gradient Norms during Training")
    plt.grid(True)
    savefig("gradient_norm", suffix=suffix)
