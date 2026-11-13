import joblib
import numpy as np
import pandas as pd
from Bio.PDB import PDBParser
import gudhi as gd

# === Function to extract C-alpha coordinates ===
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

# === Function to compute persistence diagram ===
def compute_persistence(point_cloud):
    if len(point_cloud) < 5:
        return []
    rips_complex = gd.RipsComplex(points=point_cloud, max_edge_length=10.0)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=1)
    return simplex_tree.persistence()

# === Function to summarize the persistence ===
def summarize_persistence(persistence):
    lifetimes_dim0 = [death - birth for dim, (birth, death) in persistence
                      if dim == 0 and death != float('inf')]
    lifetimes_dim1 = [death - birth for dim, (birth, death) in persistence
                      if dim == 1 and death != float('inf')]

    return {
        'num_components': len(lifetimes_dim0),
        'max_lifetime_dim0': max(lifetimes_dim0) if lifetimes_dim0 else 0,
        'mean_lifetime_dim0': np.mean(lifetimes_dim0) if lifetimes_dim0 else 0,
        'num_holes': len(lifetimes_dim1),
        'max_lifetime_dim1': max(lifetimes_dim1) if lifetimes_dim1 else 0,
        'mean_lifetime_dim1': np.mean(lifetimes_dim1) if lifetimes_dim1 else 0,
        'total_persistence': sum(lifetimes_dim1 + lifetimes_dim0),
    }

# === Load model and classify ===
def classify_pdb(pdb_path, model_path="best_xgboost_model.pkl"):
    # Load the trained model
    model = joblib.load(model_path)

    # Extract and process features
    coords = extract_ca_coordinates(pdb_path)
    persistence = compute_persistence(coords)
    if not persistence:
        print("Insufficient structure to compute persistence.")
        return

    features = summarize_persistence(persistence)
    df = pd.DataFrame([features])  # Convert to DataFrame

    # Predict label
    prediction = model.predict(df)[0]
    label = "ROTATING (1)" if prediction == 1 else "STALLED (0)"
    print(f"Prediction: {label}")

# === Example usage ===
if __name__ == "__main__":
    pdb_file = "D:/Users/admin/Documents/MANIFOLD FOR FLAGELLAR MOTORS/Pseudomonas/pdb1c7h.ent"
    classify_pdb(pdb_file)
