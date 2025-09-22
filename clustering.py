import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn_extra.cluster import KMedoids

# ====== LOAD DATA ======
df = pd.read_excel("Data_Profitability_Ratio.xlsx")

st.title("Clustering IDX30 Profitability Ratios")

# ===============================
# 0. STANDARDIZATION
# ===============================
scaler = StandardScaler()
df_std = df.copy()
df_std[["ROA","ROE","NPM","GPM"]] = scaler.fit_transform(df[["ROA","ROE","NPM","GPM"]])

# ===============================
# 1. DESCRIPTIVE ANALYSIS
# ===============================
st.header("Descriptive Analysis (Max 4 Companies)")

selected_companies = st.multiselect(
    "Pilih max 4 perusahaan:",
    df_std['Share Code'].unique(),
    max_selections=4
)

if selected_companies:
    subset = df_std[df_std['Share Code'].isin(selected_companies)]

    # BOX PLOT
    fig, ax = plt.subplots(figsize=(8, 5))
    subset_melt = subset.melt(id_vars="Share Code", 
                              value_vars=["ROA","ROE","NPM","GPM"], 
                              var_name="Rasio", value_name="Nilai")
    sns.boxplot(data=subset_melt, x="Rasio", y="Nilai", ax=ax)
    sns.stripplot(data=subset_melt, x="Rasio", y="Nilai", hue="Share Code",
                  dodge=True, ax=ax, marker="o", alpha=0.7)
    ax.legend(title="Share Code")
    st.pyplot(fig)

# ===============================
# 2. CLUSTERING K-MEDOIDS
# ===============================
st.header("Clustering K-Medoids")

X = df_std[["ROA","ROE","NPM","GPM"]].values

# hitung SSE untuk elbow method
sse = []
K = range(2, 10)
for k in K:
    model = KMedoids(n_clusters=k, random_state=42).fit(X)
    # jumlah jarak tiap titik ke medoid terdekat
    inertia = sum(
        [min([np.linalg.norm(x - medoid)**2 for medoid in model.cluster_centers_]) for x in X]
    )
    sse.append(inertia)

# plot elbow
fig, ax = plt.subplots()
ax.plot(K, sse, 'bo-')
ax.set_xlabel("Jumlah Cluster (k)")
ax.set_ylabel("SSE (Sum of Squared Errors)")
ax.set_title("Elbow Method untuk Menentukan k")
st.pyplot(fig)

# pilih k (misalnya user tentukan sendiri)
best_k = st.slider("Pilih jumlah cluster (k):", min_value=2, max_value=9, value=3)

# fit final model
model = KMedoids(n_clusters=best_k, random_state=42).fit(X)
df_std["Cluster"] = model.labels_

# PCA untuk visualisasi
pca = PCA(n_components=2)
components = pca.fit_transform(X)
df_std["PC1"], df_std["PC2"] = components[:,0], components[:,1]

fig, ax = plt.subplots(figsize=(8,6))
sns.scatterplot(data=df_std, x="PC1", y="PC2", hue="Cluster", palette="tab10", s=100)
for i, txt in enumerate(df_std["Share Code"]):
    ax.annotate(txt, (df_std["PC1"].iloc[i], df_std["PC2"].iloc[i]))
st.pyplot(fig)

# tampilkan tabel hasil clustering
st.subheader("Hasil Clustering")
st.dataframe(df_std[["Share Code","Cluster"]])

