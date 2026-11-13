import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# Assuming your data (X: scaled topological features, y: labels) is ready

# Perform PCA for dimensionality reduction to 2D
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Create a DataFrame for clearer plotting
pca_df = pd.DataFrame({
    'PCA1': X_pca[:, 0],
    'PCA2': X_pca[:, 1],
    'State Label': y.values
})

# Set Seaborn style for academic look
sns.set(style="whitegrid")

# Plot the PCA results
plt.figure(figsize=(12, 8))

# Use a clear and distinguishable color palette
sns.scatterplot(
    data=pca_df,
    x='PCA1',
    y='PCA2',
    hue='State Label',
    style='State Label',
    s=150,  # Size of markers
    palette="Set1",  # A distinct color palette
    edgecolor='black',  # Black borders for markers to enhance clarity
    linewidth=1.2    # Slightly thicker borders for better visibility
)

# Add academic titles and labels with clear fonts
plt.title('Principal Component Analysis (PCA) of Flagellar Motor Classification', fontsize=20, weight='bold')
plt.xlabel('Principal Component 1 (Explained Variance: {:.2f}%)'.format(pca.explained_variance_ratio_[0] * 100), fontsize=16)
plt.ylabel('Principal Component 2 (Explained Variance: {:.2f}%)'.format(pca.explained_variance_ratio_[1] * 100), fontsize=16)

# Adjust legend settings to look clean
plt.legend(title='Flagellar State', title_fontsize=14, fontsize=12, loc='best')

# Make gridlines subtle but present
plt.grid(True, linestyle='--', linewidth=0.7, alpha=0.8)

# Save the plot in high resolution for academic use
plt.tight_layout()
plt.savefig('flagellar_motor_pca_classification_academic.png', dpi=300)  # High-res output
plt.show()
