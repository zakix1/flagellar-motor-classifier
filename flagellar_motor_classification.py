import os
import numpy as np
import pandas as pd
from Bio.PDB import PDBParser, PPBuilder
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import gudhi as gd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier

# === Step 1: Extract C-alpha coordinates ===
def extract_ca_coordinates(pdb_file):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file)
    coords = []

    for model in structure:
        for chain in model:
            for residue in chain:
                if 'CA' in residue:
                    coords.append(residue['CA'].coord)
    return np.array(coords)

# === Step 2: Compute persistent homology (0D and 1D, optionally 2D) ===
def compute_persistence(point_cloud):
    if len(point_cloud) < 5:
        return []

    rips_complex = gd.RipsComplex(points=point_cloud, max_edge_length=10.0)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=2)  # 2D for optional future
    persistence = simplex_tree.persistence()
    return persistence

# === Step 3: Summarize persistence diagram ===
def summarize_persistence(persistence):
    lifetimes = {0: [], 1: [], 2: []}
    for dim, (birth, death) in persistence:
        if death != float('inf'):
            lifetimes[dim].append(death - birth)

    features = {
        'num_components': len(lifetimes[0]),
        'max_lifetime_dim0': max(lifetimes[0]) if lifetimes[0] else 0,
        'mean_lifetime_dim0': np.mean(lifetimes[0]) if lifetimes[0] else 0,
        'num_holes': len(lifetimes[1]),
        'max_lifetime_dim1': max(lifetimes[1]) if lifetimes[1] else 0,
        'mean_lifetime_dim1': np.mean(lifetimes[1]) if lifetimes[1] else 0,
        'total_persistence': sum(lifetimes[0] + lifetimes[1])
    }

    if lifetimes[2]:
        features['num_voids'] = len(lifetimes[2])
        features['max_lifetime_dim2'] = max(lifetimes[2])
        features['mean_lifetime_dim2'] = np.mean(lifetimes[2])
        features['total_persistence'] += sum(lifetimes[2])
    else:
        features['num_voids'] = 0
        features['max_lifetime_dim2'] = 0
        features['mean_lifetime_dim2'] = 0

    return features

# === Step 4: Compute biochemical descriptors ===
def compute_biochemical_features(pdb_file):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file)
    ppb = PPBuilder()
    seq = ''
    for pp in ppb.build_peptides(structure):
        seq += str(pp.get_sequence())

    if not seq:
        return {}

    analysis = ProteinAnalysis(seq)
    features = {
        'mol_weight': analysis.molecular_weight(),
        'aromaticity': analysis.aromaticity(),
        'instability_index': analysis.instability_index(),
        'isoelectric_point': analysis.isoelectric_point(),
        'gravy': analysis.gravy(),
        'length': len(seq)
    }

    aa_percent = analysis.get_amino_acids_percent()
    for aa, pct in aa_percent.items():
        features[f'freq_{aa}'] = pct

    return features

# === Step 5: Process dataset ===
def process_folder(folder_path, label):
    data = []

    for filename in os.listdir(folder_path):
        if filename.endswith('.ent'):
            file_path = os.path.join(folder_path, filename)
            coords = extract_ca_coordinates(file_path)
            persistence = compute_persistence(coords)

            if persistence:
                summary = summarize_persistence(persistence)
                bio_features = compute_biochemical_features(file_path)
                if bio_features:
                    summary.update(bio_features)
                summary['label'] = label
                data.append(summary)

                # Optional: Save barcode plots as images
                gd.plot_persistence_barcode(persistence)
                plt.title(filename)
                plt.savefig(f"barcodes/{filename.replace('.ent', '')}_barcode.png")
                plt.close()

    return data

# === Step 6: Main ===
if __name__ == "__main__":
    os.makedirs("barcodes", exist_ok=True)

    dataset = []
    dataset += process_folder("D:/Users/admin/Documents/MANIFOLD FOR FLAGELLAR MOTORS/Salmonella", 0)
    dataset += process_folder("D:/Users/admin/Documents/MANIFOLD FOR FLAGELLAR MOTORS/VIBRIO", 1)

    if not dataset:
        print("No data was processed.")
    else:
        df = pd.DataFrame(dataset)

        if 'label' not in df.columns:
            print("Label column missing.")
        else:
            X = df.drop(columns=["label"])
            y = df["label"]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            xgb = XGBClassifier(eval_metric='logloss', use_label_encoder=False)
            param_grid = {
                'n_estimators': [50, 100],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2]
            }

            grid_search = GridSearchCV(xgb, param_grid, cv=5, scoring='accuracy')
            grid_search.fit(X_train, y_train)

            print("\nBest Parameters:", grid_search.best_params_)

            best_model = grid_search.best_estimator_
            y_pred = best_model.predict(X_test)

            print("\nClassification Report:\n", classification_report(y_test, y_pred))
            print("Accuracy: ", accuracy_score(y_test, y_pred))

            scores = cross_val_score(best_model, X, y, cv=5)
            print("Cross-validation scores:", scores)
            print("Mean CV Accuracy:", scores.mean())
