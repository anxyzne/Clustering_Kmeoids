import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn_extra.cluster import KMedoids

# ====== LOAD DATA ======
st.sidebar.header("Pilih Dataset")

file_options = {
    "Profitability Ratio IDX30": "Data_Profitability_Ratio.xlsx"
}

selected_file = st.sidebar.selectbox("Dataset:", list(file_options.keys()))

# Load sesuai pilihan user
df = pd.read_excel(file_options[selected_file])

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
# 2. CLUSTERING (K-MEDOIDS AUTO + DBSCAN)
# ===============================
st.header("Clustering")

method = st.radio("Pilih metode clustering:", ["K-Medoids (Auto Elbow)", "DBSCAN"])
execute = st.button("Execute")

if execute:
    X = df_std[["ROA","ROE","NPM","GPM"]].values

    if method == "K-Medoids (Auto Elbow)":
        # -------------------------------
        # AUTO ELBOW K-MEDOIDS
        # -------------------------------
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

        # knee detection manual (garis lurus awal-akhir)
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

        # final model
        model = models[best_k]
        df_std["Cluster"] = model.labels_

    elif method == "DBSCAN":
        # -------------------------------
        # DBSCAN (eps auto dengan k-distance, min_samples = d+1)
        # -------------------------------
        d = X.shape[1]
        min_samples = d + 1

        neigh = NearestNeighbors(n_neighbors=min_samples)
        nbrs = neigh.fit(X)
        distances, _ = nbrs.kneighbors(X)
        distances = np.sort(distances[:, -1])

        # cari "knee" di kurva k-distance
        diffs = np.diff(distances)
        knee = np.argmax(diffs)
        eps = distances[knee]

        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(X)
        df_std["Cluster"] = labels

        # plot k-distance
        fig, ax = plt.subplots()
        ax.plot(distances)
        ax.axvline(knee, color="red", linestyle="--", label=f"Eps ≈ {eps:.2f}")
        ax.legend()
        st.pyplot(fig)

    # ===============================
    # VISUALISASI CLUSTER (PCA 2D)
    # ===============================
    pca = PCA(n_components=2)
    components = pca.fit_transform(X)
    df_std["PC1"], df_std["PC2"] = components[:,0], components[:,1]

    fig, ax = plt.subplots(figsize=(8,6))
    sns.scatterplot(data=df_std, x="PC1", y="PC2", hue="Cluster", palette="tab10", s=100)
    for i, txt in enumerate(df_std["Share Code"]):
        ax.annotate(txt, (df_std["PC1"].iloc[i], df_std["PC2"].iloc[i]))
    st.pyplot(fig)

    # ===============================
    # HASIL CLUSTERING
    # ===============================
    st.subheader("Hasil Clustering")
    st.dataframe(df_std[["Company","Share Code","Cluster"]])

    # ===============================
    # RANKING CLUSTER
    # ===============================
    cluster_means = df_std.groupby("Cluster")[["ROA","ROE","NPM","GPM"]].mean()
    cluster_means["Rata-Rata Rasio"] = cluster_means.mean(axis=1)
    ranking = cluster_means.sort_values("Rata-Rata Rasio", ascending=False)
    ranking["Ranking"] = range(1, len(ranking) + 1)

    st.subheader("Ranking Cluster berdasarkan Profitability")
    st.dataframe(ranking[["ROA","ROE","NPM","GPM","Rata-Rata Rasio","Ranking"]])

    # ===============================
    # REKOMENDASI
    # ===============================
    best_cluster = ranking.index[0]
    best_companies = df_std[df_std["Cluster"] == best_cluster]["Share Code"].tolist()

    st.success(
        f"Berdasarkan hasil clustering dengan metode **{method}**, "
        f"**Cluster {best_cluster}** memiliki rata-rata rasio profitabilitas terbaik "
        f"(Ranking 1). Disarankan untuk mempertimbangkan perusahaan dalam cluster ini untuk investasi.\n\n"
        f"Perusahaan anggota cluster terbaik: {', '.join(best_companies)}"
    )
