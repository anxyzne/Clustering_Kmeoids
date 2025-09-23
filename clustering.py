import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
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
# 2. CLUSTERING K-MEDOIDS + ELBOW AUTO
# ===============================
st.header("Clustering K-Medoids")

X = df_std[["ROA","ROE","NPM","GPM"]].values

# hitung SSE untuk elbow method
sse = []
K = range(2, 10)
models = {}
for k in K:
    model = KMedoids(n_clusters=k, random_state=42).fit(X)
    inertia = sum(
        [min([np.linalg.norm(x - medoid)**2 for medoid in model.cluster_centers_]) for x in X]
    )
    sse.append(inertia)
    models[k] = model

# metode garis lurus awal-akhir (knee detection manual)
point_first = np.array([K[0], sse[0]])
point_last = np.array([K[-1], sse[-1]])

distances = []
for i, (k, sse_val) in enumerate(zip(K, sse)):
    point = np.array([k, sse_val])
    distance = np.abs(np.cross(point_last - point_first, point_first - point)) / np.linalg.norm(point_last - point_first)
    distances.append(distance)

best_k = K[int(np.argmax(distances))]

# plot elbow
fig, ax = plt.subplots()
ax.plot(K, sse, 'bo-')
ax.axvline(best_k, color='red', linestyle='--', label=f"Optimal k = {best_k}")
ax.set_xlabel("Jumlah Cluster (k)")
ax.set_ylabel("SSE (Sum of Squared Errors)")
ax.set_title("Elbow Method dengan Knee Detection")
ax.legend()
st.pyplot(fig)

# fit final model
model = models[best_k]
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
st.dataframe(df_std[["Company","Cluster"]])

# ===============================
# 3. RATA-RATA PROFITABILITY PER CLUSTER
# ===============================
st.subheader("Rata-rata Profitability per Cluster")

cluster_means = df_std.groupby("Cluster")[["ROA","ROE","NPM","GPM"]].mean()

fig, ax = plt.subplots(figsize=(8,5))
cluster_means.plot(kind="bar", ax=ax)
plt.xticks(rotation=0)
plt.title("Rata-rata Profitability per Cluster")
plt.ylabel("Rata-rata Standarisasi Nilai")
st.pyplot(fig)

st.dataframe(cluster_means)

# ===============================
# 4. RANKING CLUSTER
# ===============================
cluster_means["Cluster_Score"] = cluster_means.mean(axis=1)
ranking = cluster_means.sort_values("Cluster_Score", ascending=False)
ranking["Ranking"] = range(1, len(ranking) + 1)

st.subheader("Ranking Cluster berdasarkan Profitability")
st.dataframe(ranking[["ROA","ROE","NPM","GPM","Cluster_Score","Ranking"]])

# ===============================
# 5. REKOMENDASI
# ===============================
best_cluster = ranking.index[0]
best_companies = df_std[df_std["Cluster"] == best_cluster]["Share Code"].tolist()

st.success(
    f"Berdasarkan hasil clustering, **Cluster {best_cluster}** memiliki rata-rata rasio profitabilitas terbaik "
    f"(Ranking 1). Disarankan untuk mempertimbangkan perusahaan dalam cluster ini untuk investasi.\n\n"
    f"Perusahaan anggota cluster terbaik: {', '.join(best_companies)}"
)

