from typing import Optional, Tuple

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import umap
from loguru import logger


def embedding_plot(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    label_dict: Optional[dict] = None,
    figsize: Tuple[int, int] = (10, 5),
    save_path: Optional[str] = None,
    return_figure: bool = False,
    max_samples: int = 5_000,
    debug: bool = False,
):
    # Subsample image embeddings if more than X samples (stratified by class)
    if y is not None and X.shape[0] > max_samples:
        logger.debug(
            f"Embeddings exceed {max_samples} samples: {X.shape[0]}. Subsampling..."
        )
        unique_labels, counts = np.unique(y, return_counts=True)
        total_samples = X.shape[0]
        subsampled_indices = []

        for lab, count in zip(unique_labels, counts):
            # Find indices for this class
            class_indices = np.where(y == lab)[0]
            # Compute number of samples for this class (proportional to its frequency)
            n_samples = int(np.round((count / total_samples) * max_samples))
            # Ensure at least one sample is taken if that class is present
            if n_samples == 0 and count > 0:
                n_samples = 1
            chosen = np.random.choice(class_indices, size=n_samples, replace=False)
            subsampled_indices.extend(chosen)
        subsampled_indices = np.array(subsampled_indices)
        X = X[subsampled_indices]
        y = y[subsampled_indices]
    elif X.shape[0] > max_samples:
        logger.debug(
            f"Embeddings exceed {max_samples} samples: {X.shape[0]}. Subsampling..."
        )
        indices = np.random.choice(
            np.arange(X.shape[0]), size=max_samples, replace=False
        )
        X = X[indices]

    if X.shape[0] > 2:
        # apply if the dimension is larger than 2
        umap_transformer = umap.UMAP(
            n_components=2, n_neighbors=100, random_state=42, n_jobs=1
        )
        X = umap_transformer.fit_transform(X)
    x_min, x_max = np.min(X, axis=0), np.max(X, axis=0)
    X = (X - x_min) / (x_max - x_min)
    plt.figure(figsize=figsize)
    fig, ax = plt.subplots(1, 1)
    if y is not None:
        if label_dict is not None:
            colors = cm.rainbow(np.linspace(0, 1, len(set(y))))
            for id_cls, color in zip(set(y), colors):
                cls_idx = np.where(y == id_cls)[0]
                id_cls = label_dict.get(id_cls, id_cls)
                ax.scatter(
                    X[cls_idx, 0],
                    X[cls_idx, 1],
                    label=id_cls,
                    color=color,
                    s=20,
                    alpha=0.5,
                )
            plt.legend()
        else:
            scatterplot = ax.scatter(
                X[:, 0],
                X[:, 1],
                c=y,
                cmap="viridis",
                s=20,
                alpha=0.5,
            )
            plt.colorbar(scatterplot, ax=ax)
    else:
        ax.scatter(X[:, 0], X[:, 1], alpha=0.7, s=20)
    plt.xticks([]), plt.yticks([])
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
    if debug:
        plt.show()
    if return_figure:
        return fig


def embedding_plot_w_markers(
    X: np.ndarray,
    y: np.ndarray = None,
    label_dict: dict = None,
    figsize: tuple = (10, 5),
    text_X: np.ndarray = None,
    text_y: np.ndarray = None,
    text_marker: str = "^",
    text_legend_label: str = "Text",
    max_samples: int = 5_000,
    debug: bool = False,
):
    # Subsample image embeddings if more than X samples (stratified by class)
    if y is not None and X.shape[0] > max_samples:
        logger.debug(
            f"Embeddings exceed {max_samples} samples: {X.shape[0]}. Subsampling..."
        )
        unique_labels, counts = np.unique(y, return_counts=True)
        total_samples = X.shape[0]
        subsampled_indices = []

        for lab, count in zip(unique_labels, counts):
            # Find indices for this class
            class_indices = np.where(y == lab)[0]
            # Compute number of samples for this class (proportional to its frequency)
            n_samples = int(np.round((count / total_samples) * max_samples))
            # Ensure at least one sample is taken if that class is present
            if n_samples == 0 and count > 0:
                n_samples = 1
            chosen = np.random.choice(class_indices, size=n_samples, replace=False)
            subsampled_indices.extend(chosen)
        subsampled_indices = np.array(subsampled_indices)
        X = X[subsampled_indices]
        y = y[subsampled_indices]
    elif X.shape[0] > max_samples:
        logger.debug(
            f"Embeddings exceed {max_samples} samples: {X.shape[0]}. Subsampling..."
        )
        indices = np.random.choice(
            np.arange(X.shape[0]), size=max_samples, replace=False
        )
        X = X[indices]

    if X.shape[0] > 2:
        # apply if the dimension is larger than 2
        if text_X is not None:
            X = np.concatenate([text_X, X])
        umap_transformer = umap.UMAP(
            n_components=2, n_neighbors=100, random_state=42, n_jobs=1
        )
        X = umap_transformer.fit_transform(X)
    x_min, x_max = np.min(X, axis=0), np.max(X, axis=0)
    X = (X - x_min) / (x_max - x_min)
    if text_X is not None:
        text_X = X[: len(text_X)]
        X = X[len(text_X) :]
    plt.figure(figsize=figsize)
    fig, ax = plt.subplots(1, 1)

    # If no labels given, just do a single scatter
    if y is None:
        scatterplot = ax.scatter(
            X[:, 0],
            X[:, 1],
            c=y,
            cmap="viridis",
            s=20,
            alpha=0.5,
        )
        plt.colorbar(scatterplot, ax=ax)
    else:
        unique_labels = np.unique(y)
        cmap = plt.cm.get_cmap("tab10", len(unique_labels))

        # Plot the image embeddings by class
        for i, lab in enumerate(unique_labels):
            idx = y == lab
            color = cmap(i)
            if label_dict is not None and lab in label_dict:
                class_label = label_dict[lab]
            else:
                class_label = f"Class {lab}"
            plt.scatter(
                X[idx, 0],
                X[idx, 1],
                marker="o",
                color=color,
                label=class_label,
                s=20,
                alpha=0.5,
            )

        # Plot text embeddings if provided
        if text_X is not None and text_y is not None:
            # Plot each class's text points with the same color but different marker
            for i, lab in enumerate(unique_labels):
                idx = text_y == lab
                color = cmap(i)
                plt.scatter(
                    text_X[idx, 0],
                    text_X[idx, 1],
                    marker=text_marker,
                    color=color,
                    alpha=0.9,
                )

            # Add one dummy scatter for text marker to appear in legend
            text_handle = plt.scatter([], [], marker=text_marker, color="black")
            # Combine with existing legend
            handles, labels = plt.gca().get_legend_handles_labels()
            # Append our single text marker entry
            handles.append(text_handle)
            labels.append(text_legend_label)
            plt.legend(handles, labels, loc="best")
        else:
            plt.legend(loc="best")
    return fig
