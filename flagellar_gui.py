import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import numpy as np
import pandas as pd
from Bio.PDB import PDBParser
from Bio.PDB.PDBExceptions import PDBConstructionException
import gudhi as gd
import joblib

# === Load trained model ===
model = joblib.load(Path(__file__).resolve().with_name("best_xgboost_model.pkl"))

# === Feature extraction as before ===
def extract_ca_coordinates(pdb_file):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file)
    coords = [residue['CA'].coord for model in structure for chain in model
              for residue in chain if 'CA' in residue]
    return np.array(coords)

def compute_persistence(point_cloud):
    if len(point_cloud) < 5:
        return []
    rips = gd.RipsComplex(points=point_cloud, max_edge_length=10.0)
    st = rips.create_simplex_tree(max_dimension=1)
    st.persistence()
    return st

def summarize_persistence(st):
    persistence = st.persistence()
    lifetimes_dim0 = [d - b for dim, (b, d) in persistence if dim == 0 and d != float('inf')]
    lifetimes_dim1 = [d - b for dim, (b, d) in persistence if dim == 1 and d != float('inf')]
    return {
        'num_components': len(lifetimes_dim0),
        'max_lifetime_dim0': max(lifetimes_dim0) if lifetimes_dim0 else 0,
        'mean_lifetime_dim0': np.mean(lifetimes_dim0) if lifetimes_dim0 else 0,
        'num_holes': len(lifetimes_dim1),
        'max_lifetime_dim1': max(lifetimes_dim1) if lifetimes_dim1 else 0,
        'mean_lifetime_dim1': np.mean(lifetimes_dim1) if lifetimes_dim1 else 0,
        'total_persistence': sum(lifetimes_dim1 + lifetimes_dim0),
    }

# === GUI logic ===
def classify_protein():
    file_path = filedialog.askopenfilename(filetypes=[("PDB files", "*.ent *.pdb")])
    if not file_path:
        return

    try:
        coords = extract_ca_coordinates(file_path)
        if len(coords) < 5:
            raise ValueError("The selected structure must contain at least 5 C-alpha atoms.")

        st = compute_persistence(coords)
        features = summarize_persistence(st)
        df = pd.DataFrame([features])
        prediction = model.predict(df)[0]
    except (OSError, ValueError, PDBConstructionException) as exc:
        messagebox.showerror("Classification failed", str(exc))
        return

    result = "Rotating" if prediction == 1 else "Stalled"
    result_label.config(text=f"Prediction: {result}")

    plot_barcode(st)

def plot_barcode(simplex_tree):
    fig, ax = plt.subplots(figsize=(6, 3))
    gd.plot_persistence_barcode(simplex_tree.persistence(), axes=ax)
    ax.set_title("Persistence Barcode")
    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas.get_tk_widget().pack()
    canvas.draw()

# === GUI setup ===
window = tk.Tk()
window.title("Flagellar Motor Classifier")
window.geometry("700x600")

btn = tk.Button(window, text="Select .ent File and Classify", command=classify_protein)
btn.pack(pady=20)

result_label = tk.Label(window, text="Prediction: ", font=("Arial", 16))
result_label.pack(pady=10)

window.mainloop()
