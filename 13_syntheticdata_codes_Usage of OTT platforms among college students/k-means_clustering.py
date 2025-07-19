import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Step 1: Load preprocessed dataset
df = pd.read_csv("OTT_Survey_50_Responses_Preprocessed.csv")

# Step 2: Drop target and ID columns (optional)
df_features = df.drop(columns=[col for col in df.columns if "Affects Academics" in col or col == "ID"])

# Step 3: Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_features)

# Step 4: Apply K-Means
k = 3  # You can change this
kmeans = KMeans(n_clusters=k, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# Step 5: Dimensionality Reduction (PCA for 2D visualization)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Step 6: Plot the clusters
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', s=50)
plt.title(f'K-Means Clustering on OTT Survey Data')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar(label='Cluster')
plt.grid(True)
plt.tight_layout()
plt.show()
