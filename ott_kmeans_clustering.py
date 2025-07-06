import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load preprocessed dataset
df = pd.read_csv("OTT_Usage_Survey_Preprocessed.csv")

# Drop target column if present (this is unsupervised)
df_cluster = df.drop(columns=["Do you plan to continue using OTT platforms in the future?"], errors='ignore')

# Identify categorical columns
categorical_cols = df_cluster.select_dtypes(include='object').columns.tolist()

# One-hot encode categorical features
encoder = OneHotEncoder(sparse_output=False)
encoded_cats = encoder.fit_transform(df_cluster[categorical_cols])
encoded_cat_df = pd.DataFrame(encoded_cats, columns=encoder.get_feature_names_out(categorical_cols))

# Combine with numerical data
numerical_df = df_cluster.drop(columns=categorical_cols)
final_df = pd.concat([encoded_cat_df, numerical_df.reset_index(drop=True)], axis=1)

# Apply KMeans clustering (try 3 clusters)
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(final_df)
final_df['Cluster'] = clusters

# Reduce to 2D for visualization using PCA
pca = PCA(n_components=2)
reduced = pca.fit_transform(final_df.drop(columns=["Cluster"]))

# Plot the clusters
plt.figure(figsize=(8, 6))
plt.scatter(reduced[:, 0], reduced[:, 1], c=clusters, cmap='viridis', s=50)
plt.title("K-Means Clustering of OTT Users")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.colorbar(label="Cluster")
plt.grid(True)
plt.tight_layout()
plt.show()
