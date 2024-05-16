# Import necessary libraries
import pandas as pd
from Bio import Entrez
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Set email for NCBI Entrez
Entrez.email = "your-email@example.com"

def fetch_sra_metadata(sra_id):
    """Fetch metadata for a given SRA ID using Entrez utilities."""
    handle = Entrez.esummary(db="sra", id=sra_id)
    record = Entrez.read(handle)
    handle.close()
    return record

def simple_pca_analysis(data):
    """Perform PCA on given dataset and plot the first two principal components."""
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(data)
    principal_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
    plt.figure(figsize=(8,6))
    plt.scatter(principal_df['PC1'], principal_df['PC2'])
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA of SRA Data')
    plt.show()

# Example SRA ID, replace with relevant ID
sra_id = 'SRX000000'  # Example, use a valid SRA ID
metadata = fetch_sra_metadata(sra_id)
print(metadata)

# Assume 'data' is loaded as a DataFrame with relevant genomic features
# For example purposes, let's create a dummy dataset
data = pd.DataFrame({
    'feature1': [0.1, 0.2, 0.4, 0.5],
    'feature2': [0.6, 0.7, 0.8, 0.9]
})

# Perform PCA on the dataset
simple_pca_analysis(data)