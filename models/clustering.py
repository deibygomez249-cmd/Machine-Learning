import os
import io
import base64
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

DATA_PATH = os.path.join('data', 'used_cars_clean.csv')
REFERENCE_YEAR = 2026
FEATURES = ['vehicle_age', 'mileage']
FEATURE_LABELS = {
    'vehicle_age': 'Vehicle Age (years)',
    'mileage': 'Mileage (miles)'
}
RANDOM_STATE = 42
N_CLUSTERS = 3
MANUAL_SIZE = 100


def load_clean_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            'Used-car clean dataset not found. Run data/process_used_cars.py first.'
        )
    df = pd.read_csv(DATA_PATH)
    return df.dropna(subset=FEATURES).reset_index(drop=True)


def get_dataset_info():
    df = load_clean_data()
    return {
        'records': int(len(df)),
        'features': FEATURES,
        'feature_labels': FEATURE_LABELS,
        'reference_year': REFERENCE_YEAR
    }


def select_manual_sample(df, size=MANUAL_SIZE):
    if len(df) < size:
        raise ValueError('Dataset has fewer than 100 valid records.')
    return df.sample(n=size, random_state=RANDOM_STATE).reset_index(drop=True)


def euclidean_distance(point, centroid):
    return float(np.sqrt(np.sum((point - centroid) ** 2)))


def compute_iteration(scaled_data, centroids):
    distances = np.zeros((len(scaled_data), len(centroids)))
    for i, point in enumerate(scaled_data):
        for j, centroid in enumerate(centroids):
            distances[i, j] = euclidean_distance(point, centroid)

    assignments = np.argmin(distances, axis=1)

    new_centroids = []
    for k in range(len(centroids)):
        members = scaled_data[assignments == k]
        if len(members) == 0:
            new_centroids.append(centroids[k].copy())
        else:
            new_centroids.append(members.mean(axis=0))
    new_centroids = np.array(new_centroids)

    return distances, assignments, new_centroids


def within_cluster_variance(scaled_data, assignments, centroids):
    total = 0.0
    for k in range(len(centroids)):
        members = scaled_data[assignments == k]
        if len(members) == 0:
            continue
        total += float(np.sum((members - centroids[k]) ** 2))
    return total


def run_manual_kmeans(df_sample, iterations=3):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df_sample[FEATURES])

    centroid_indices = [0, len(df_sample) // 2, len(df_sample) - 1]
    initial_centroids = scaled[centroid_indices].copy()

    history = []
    current_centroids = initial_centroids

    for it in range(1, iterations + 1):
        distances, assignments, new_centroids = compute_iteration(scaled, current_centroids)
        variance = within_cluster_variance(scaled, assignments, new_centroids)

        history.append({
            'iteration': it,
            'centroids_before': current_centroids.tolist(),
            'distances': distances.tolist(),
            'assignments': assignments.tolist(),
            'centroids_after': new_centroids.tolist(),
            'variance': round(variance, 4)
        })

        current_centroids = new_centroids

    return {
        'scaled_data': scaled.tolist(),
        'scaler_mean': scaler.mean_.tolist(),
        'scaler_scale': scaler.scale_.tolist(),
        'initial_centroids': initial_centroids.tolist(),
        'centroid_indices': centroid_indices,
        'iterations': history,
        'sample': df_sample.to_dict(orient='records'),
        'reference_year': REFERENCE_YEAR
    }


def fit_sklearn_kmeans(df):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[FEATURES])

    model = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=10)
    labels = model.fit_predict(scaled)

    centroids_scaled = model.cluster_centers_
    centroids_original = scaler.inverse_transform(centroids_scaled)

    df_result = df.copy()
    df_result['cluster'] = labels

    silhouette = float(silhouette_score(scaled, labels))
    inertia = float(model.inertia_)

    summary = []
    for k in range(N_CLUSTERS):
        members = df_result[df_result['cluster'] == k]
        summary.append({
            'cluster': k,
            'size': int(len(members)),
            'avg_vehicle_age': round(float(members['vehicle_age'].mean()), 2),
            'avg_mileage': round(float(members['mileage'].mean()), 2),
            'centroid_vehicle_age': round(float(centroids_original[k][0]), 2),
            'centroid_mileage': round(float(centroids_original[k][1]), 2)
        })

    return {
        'scaled_data': scaled,
        'labels': labels,
        'centroids_scaled': centroids_scaled,
        'centroids_original': centroids_original,
        'df_result': df_result,
        'summary': summary,
        'silhouette': round(silhouette, 4),
        'inertia': round(inertia, 4),
        'records': int(len(df_result))
    }


def interpret_clusters(summary):
    sorted_clusters = sorted(summary, key=lambda c: c['centroid_vehicle_age'])
    interpretations = []
    for rank, cluster in enumerate(sorted_clusters):
        if rank == 0:
            profile = 'Newer vehicles with lower accumulated mileage'
        elif rank == len(sorted_clusters) - 1:
            profile = 'Older vehicles with higher accumulated mileage'
        else:
            profile = 'Intermediate vehicles in age and mileage'
        interpretations.append({
            'cluster': cluster['cluster'],
            'profile': profile,
            'size': cluster['size'],
            'avg_vehicle_age': cluster['avg_vehicle_age'],
            'avg_mileage': cluster['avg_mileage']
        })
    return interpretations


def plot_manual_iteration(df_sample, assignments, centroids_scaled, scaler, iteration):
    df = df_sample.copy()
    df['cluster'] = assignments

    centroids_original = scaler.inverse_transform(centroids_scaled)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#0d6efd', '#198754', '#dc3545']

    for k in range(len(centroids_scaled)):
        members = df[df['cluster'] == k]
        if len(members) > 0:
            ax.scatter(members['vehicle_age'], members['mileage'],
                       color=colors[k], alpha=0.6, s=60,
                       label=f'Cluster {k} (n={len(members)})',
                       edgecolors='white')

        ax.scatter(centroids_original[k][0], centroids_original[k][1],
                   color=colors[k], marker='X', s=300,
                   edgecolors='black', linewidths=2,
                   label=f'Centroid {k}')

    ax.set_xlabel('Vehicle Age (years)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mileage (miles)', fontsize=12, fontweight='bold')
    ax.set_title(f'Manual K-Means - Iteration {iteration}', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=100, bbox_inches='tight', facecolor='white')
    img.seek(0)
    plt.close()
    return base64.b64encode(img.getvalue()).decode()


def plot_sklearn_clusters(df_result, centroids_original):
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#0d6efd', '#198754', '#dc3545']

    for k in range(N_CLUSTERS):
        members = df_result[df_result['cluster'] == k]
        ax.scatter(members['vehicle_age'], members['mileage'],
                   color=colors[k], alpha=0.5, s=20,
                   label=f'Cluster {k} (n={len(members)})')
        ax.scatter(centroids_original[k][0], centroids_original[k][1],
                   color=colors[k], marker='X', s=350,
                   edgecolors='black', linewidths=2.5,
                   label=f'Centroid {k}')

    ax.set_xlabel('Vehicle Age (years)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mileage (miles)', fontsize=12, fontweight='bold')
    ax.set_title('Used Car Market Segmentation - K-Means Clustering',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=100, bbox_inches='tight', facecolor='white')
    img.seek(0)
    plt.close()
    return base64.b64encode(img.getvalue()).decode()


def plot_variance(history):
    iterations = [h['iteration'] for h in history]
    variances = [h['variance'] for h in history]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(iterations, variances, marker='o', color='#0d6efd',
            linewidth=2.5, markersize=10)
    ax.set_xlabel('Iteration', fontsize=12, fontweight='bold')
    ax.set_ylabel('Within-Cluster Variance', fontsize=12, fontweight='bold')
    ax.set_title('Within-Cluster Variance per Iteration', fontsize=13, fontweight='bold')
    ax.set_xticks(iterations)
    ax.grid(True, alpha=0.3)

    for x, y in zip(iterations, variances):
        ax.annotate(f'{y:.3f}', (x, y), textcoords='offset points',
                    xytext=(0, 10), ha='center')

    plt.tight_layout()
    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=100, bbox_inches='tight', facecolor='white')
    img.seek(0)
    plt.close()
    return base64.b64encode(img.getvalue()).decode()