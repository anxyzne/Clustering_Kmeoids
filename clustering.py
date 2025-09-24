import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn_extra.cluster import KMedoids
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from kneed import KneeLocator



# ====== LOAD DATA ======
st.sidebar.header("Pilih Dataset")

file_options = {
    "Profitability Ratio IDX30": "Data_Profitability_Ratio.xlsx"
}

selected_file = st.sidebar.selectbox("Dataset:", list(file_options.keys()))

# Load sesuai pilihan user
df = pd.read_excel(file_options[selected_file])

st.title("Clustering Profitability Ratios")

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
st.header("Clustering Method")

method = st.radio("Pilih metode clustering:", ["K-Medoids", "DBSCAN"])
execute = st.button("Execute")

if execute:
    X = df_std[["ROA","ROE","NPM","GPM"]].values

    if method == "K-Medoids":
        # -------------------------------
        # K-MEDOIDS + ELBOW AUTO
        # -------------------------------
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
    
        # knee detection manual
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
        ax.set_ylabel("SSE")
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
    
        # hasil clustering
        st.subheader("Hasil Clustering")
        st.dataframe(df_std[["Company","Cluster"]])
    
        # ranking cluster
        cluster_means = df_std.groupby("Cluster")[["ROA","ROE","NPM","GPM"]].mean()
        cluster_means["Rata-Rata Rasio"] = cluster_means.mean(axis=1)
        ranking = cluster_means.sort_values("Rata-Rata Rasio", ascending=False)
        ranking["Ranking"] = range(1, len(ranking) + 1)
    
        st.subheader("Ranking Cluster berdasarkan Profitability")
        st.dataframe(ranking)
    
        best_cluster = ranking.index[0]
        best_companies = df_std[df_std["Cluster"] == best_cluster]["Share Code"].tolist()
        st.success(
            f"Berdasarkan hasil clustering, **Cluster {best_cluster}** memiliki rata-rata profitabilitas terbaik "
            f"(Ranking 1). Disarankan untuk mempertimbangkan perusahaan dalam cluster ini untuk investasi.\n\n"
            f"Perusahaan anggota cluster terbaik: {', '.join(best_companies)}"
        )

    elif method == "DBSCAN":
        # -------------------------------
        # DBSCAN (eps auto dengan k-distance, min_samples = d+1)
        # -------------------------------
        st.header("Clustering DBSCAN")
    
        X = df_std[["ROA","ROE","NPM","GPM"]].values
        d = X.shape[1]
        min_samples = d + 1
        k = min_samples - 1
    
        neigh = NearestNeighbors(n_neighbors=k)
        neigh.fit(X)
        distances, _ = neigh.kneighbors(X)
        k_distances = np.sort(distances[:, -1])
    
        # KneeLocator untuk eps
        from kneed import KneeLocator
        kneedle = KneeLocator(range(len(k_distances)), k_distances, curve='convex', direction='increasing')
        eps = k_distances[kneedle.knee]
    
        # plot k-distance
        fig, ax = plt.subplots(figsize=(8,5))
        ax.plot(k_distances, label='k-distance')
        ax.axvline(x=kneedle.knee, color='r', linestyle='--', label=f"Index = {kneedle.knee}")
        ax.axhline(y=eps, color='g', linestyle='--', label=f"Epsilon ≈ {eps:.4f}")
        ax.set_xlabel("Data Point Index (sorted)")
        ax.set_ylabel(f"{k}-Distance")
        ax.set_title("DBSCAN k-Distance Plot")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)
    
        # fit DBSCAN
        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(X)
        df_std["Cluster"] = labels
    
        # PCA untuk visualisasi
        pca = PCA(n_components=2)
        components = pca.fit_transform(X)
        df_std["PC1"], df_std["PC2"] = components[:,0], components[:,1]
    
        fig, ax = plt.subplots(figsize=(8,6))
        sns.scatterplot(data=df_std, x="PC1", y="PC2", hue="Cluster", palette="tab10", s=100)
        for i, txt in enumerate(df_std["Share Code"]):
            ax.annotate(txt, (df_std["PC1"].iloc[i], df_std["PC2"].iloc[i]))
        st.pyplot(fig)

        # hasil clustering
        st.subheader("Hasil Clustering")
        st.dataframe(df_std[["Company","Cluster"]])
    
        # rekomendasi DBSCAN
        if -1 in df_std["Cluster"].unique():
            outliers = df_std[df_std["Cluster"] == -1]["Share Code"].tolist()
            st.warning(
                f"DBSCAN mendeteksi perusahaan berikut sebagai **outlier** (Cluster -1): {', '.join(outliers)}. "
                f"Perusahaan ini memiliki rasio profitabilitas yang jauh berbeda dari lainnya, "
                f"sehingga perlu perhatian lebih atau analisis lebih dalam sebelum dipertimbangkan untuk investasi."
            )
    
        valid_clusters = [c for c in df_std["Cluster"].unique() if c != -1]
        if valid_clusters:
            cluster_means = df_std[df_std["Cluster"] != -1].groupby("Cluster")[["ROA","ROE","NPM","GPM"]].mean()
            cluster_means["Rata-Rata Rasio"] = cluster_means.mean(axis=1)
            ranking = cluster_means.sort_values("Rata-Rata Rasio", ascending=False)
            ranking["Ranking"] = range(1, len(ranking) + 1)
    
            st.subheader("Ranking Cluster (tanpa outlier -1)")
            st.dataframe(ranking)
    
            best_cluster = ranking.index[0]
            best_companies = df_std[df_std["Cluster"] == best_cluster]["Share Code"].tolist()
            st.success(
                f"Cluster {best_cluster} (Ranking 1) memiliki rata-rata profitabilitas terbaik. "
                f"Perusahaan anggota cluster ini disarankan untuk investasi: {', '.join(best_companies)}."
            )






