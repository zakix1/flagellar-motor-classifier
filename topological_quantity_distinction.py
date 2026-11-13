
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Bio.PDB import PDBParser
from ripser import ripser
from scipy.spatial.distance import pdist
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.decomposition import PCA


# ============================================================
# 1. Extract protein C-alpha coordinates
# ============================================================

def load_protein(path):

    structure = PDBParser(QUIET=True).get_structure("P", path)
    model = next(structure.get_models())

    coordinates = []

    for chain in model:
        for residue in chain:

            if residue.id[0] == " " and "CA" in residue:
                coordinates.append(
                    residue["CA"].coord.astype(float)
                )

    X = np.asarray(coordinates)

    if len(X) < 4:
        raise ValueError("Insufficient C-alpha coordinates.")

    return X


# ============================================================
# 2. Geometric descriptors
# ============================================================

def geometric_features(X):

    distances = pdist(X)

    centroid = X.mean(axis=0)
    centered = X - centroid

    radius_gyration = np.sqrt(
        np.mean(np.sum(centered**2, axis=1))
    )

    covariance = np.cov(centered.T)
    eigenvalues = np.linalg.eigvalsh(covariance)

    eigenvalues = np.maximum(eigenvalues, 0)

    return np.array([
        len(X),
        distances.mean(),
        distances.std(),
        np.median(distances),
        np.percentile(distances, 25),
        np.percentile(distances, 75),
        radius_gyration,
        eigenvalues[-1],
        eigenvalues[-2],
        eigenvalues[-3]
    ])


# ============================================================
# 3. Persistent homology descriptors
# ============================================================

def topological_features(X, maxdim=2, threshold=15.0):

    result = ripser(
        X,
        maxdim=maxdim,
        thresh=threshold
    )

    diagrams = result["dgms"]

    features = []

    for k in range(maxdim + 1):

        diagram = diagrams[k]

        # Exclude infinite death times
        finite = diagram[
            np.isfinite(diagram[:, 1])
        ]

        if len(finite) > 0:

            lifetimes = (
                finite[:, 1] - finite[:, 0]
            )

            features.extend([
                len(lifetimes),
                lifetimes.sum(),
                lifetimes.mean(),
                lifetimes.max(),
                np.sum(lifetimes**2)
            ])

        else:

            features.extend([0, 0, 0, 0, 0])

    return np.asarray(features), diagrams


# ============================================================
# 4. Process the 50-protein dataset
# ============================================================

def process_dataset(input_directory):

    files = sorted(
        glob.glob(os.path.join(input_directory, "*.pdb"))
        + glob.glob(os.path.join(input_directory, "*.PDB"))
    )

    if not files:
        raise RuntimeError("No PDB files found.")

    geometry_data = []
    topology_data = []
    names = []

    diagrams = {}

    for path in files:

        name = os.path.basename(path)

        print("Processing:", name)

        X = load_protein(path)

        G = geometric_features(X)

        T, D = topological_features(
            X,
            maxdim=2,
            threshold=15.0
        )

        names.append(name)
        geometry_data.append(G)
        topology_data.append(T)

        diagrams[name] = D

        print(
            "Residues:", len(X),
            "| Geometric dimension:", len(G),
            "| Topological dimension:", len(T)
        )

    G = np.asarray(geometry_data)
    T = np.asarray(topology_data)

    return names, G, T, diagrams


# ============================================================
# 5. Construct separate RBF kernels
# ============================================================

def construct_kernels(G, T):

    scaler_G = StandardScaler()
    scaler_T = StandardScaler()

    G_scaled = scaler_G.fit_transform(G)
    T_scaled = scaler_T.fit_transform(T)

    K_geometry = rbf_kernel(
        G_scaled,
        gamma=1.0 / G_scaled.shape[1]
    )

    K_topology = rbf_kernel(
        T_scaled,
        gamma=1.0 / T_scaled.shape[1]
    )

    return (
        K_geometry,
        K_topology,
        G_scaled,
        T_scaled
    )


# ============================================================
# 6. Compare geometric and topological representations
# ============================================================

def compare_kernels(KG, KT):

    # Center the kernel matrices
    n = KG.shape[0]

    H = np.eye(n) - np.ones((n, n)) / n

    KGc = H @ KG @ H
    KTc = H @ KT @ H

    # Centered kernel alignment
    alignment = np.sum(KGc * KTc) / (
        np.linalg.norm(KGc, "fro")
        * np.linalg.norm(KTc, "fro")
        + 1e-12
    )

    print("\nGEOMETRY-TOPOLOGY COMPARISON")
    print("----------------------------")
    print("Centered kernel alignment:", alignment)

    return alignment


# ============================================================
# 7. Visualization
# ============================================================

def plot_kernels(KG, KT, names, output_directory):

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].imshow(KG, vmin=0, vmax=1)
    axes[0].set_title("Geometric RBF Kernel")
    axes[0].set_xlabel("Protein")
    axes[0].set_ylabel("Protein")

    axes[1].imshow(KT, vmin=0, vmax=1)
    axes[1].set_title("Topological RBF Kernel")
    axes[1].set_xlabel("Protein")
    axes[1].set_ylabel("Protein")

    plt.tight_layout()

    plt.savefig(
        os.path.join(output_directory, "kernel_comparison.png"),
        dpi=200
    )

    plt.close()


def plot_pca(G, T, output_directory):

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, data, title in [
        (axes[0], G, "Geometric feature space"),
        (axes[1], T, "Persistent homology feature space")
    ]:

        coordinates = PCA(
            n_components=2
        ).fit_transform(data)

        ax.scatter(
            coordinates[:, 0],
            coordinates[:, 1]
        )

        ax.set_title(title)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")

    plt.tight_layout()

    plt.savefig(
        os.path.join(output_directory, "pca_comparison.png"),
        dpi=200
    )

    plt.close()


# ============================================================
# 8. Main execution
# ============================================================

def main():

    input_directory = "./pdbs"
    output_directory = "./results"

    os.makedirs(output_directory, exist_ok=True)

    names, G, T, diagrams = process_dataset(
        input_directory
    )

    KG, KT, Gs, Ts = construct_kernels(G, T)

    alignment = compare_kernels(KG, KT)

    plot_kernels(
        KG, KT, names, output_directory
    )

    plot_pca(
        Gs, Ts, output_directory
    )

    # Save feature datasets
    pd.DataFrame(G).to_csv(
        os.path.join(output_directory, "geometry_features.csv"),
        index=False
    )

    pd.DataFrame(T).to_csv(
        os.path.join(output_directory, "topology_features.csv"),
        index=False
    )

    pd.DataFrame({
        "protein": names,
        "geometry_topology_alignment": [
            alignment
        ] * len(names)
    }).to_csv(
        os.path.join(output_directory, "comparison.csv"),
        index=False
    )

    np.save(
        os.path.join(output_directory, "geometry_kernel.npy"),
        KG
    )

    np.save(
        os.path.join(output_directory, "topology_kernel.npy"),
        KT
    )

    print("\nAnalysis completed.")
    print("Number of proteins:", len(names))
    print("Results directory:", output_directory)


if __name__ == "__main__":
    main()
