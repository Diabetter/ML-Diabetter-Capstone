import pandas as pd
import numpy as np
import random
from itertools import combinations
from sklearn.cluster import KMeans
import os
import sys
import warnings
import argparse
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--berat-badan')
parser.add_argument('--tinggi')
parser.add_argument('--usia')
parser.add_argument('--jenis-kelamin')
parser.add_argument('--aktivitas')
args = parser.parse_args()

# berat_badan = 60
# tinggi = 176
# usia = 21
# jenis_kelamin = 'pria'
# aktivitas = 'sedang'

berat_badan = int(args.berat_badan)
tinggi = int(args.tinggi)
usia = int(args.usia)
jenis_kelamin = args.jenis_kelamin
aktivitas = args.aktivitas
filter = 4

def resource_path(relative):
    #print(os.environ)
    application_path = os.path.abspath(".")
    if getattr(sys, 'frozen', False):
        # If the application is run as a bundle, the pyInstaller bootloader
        # extends the sys module by a flag frozen=True and sets the app 
        # path into variable _MEIPASS'.
        application_path = sys._MEIPASS
    #print(application_path)
    return os.path.join(application_path, relative)

def hitung_akg_diabetes(berat_badan, tinggi, usia, jenis_kelamin):
    # Menghitung IMT
    tinggi_m = tinggi / 100
    imt = berat_badan / (tinggi_m ** 2)
    
    # Menghitung BBI (Berat Badan Ideal)
    bbi = (tinggi - 100) - 0.1 * (tinggi - 100)
    
    # Menghitung Kalori Basal
    if jenis_kelamin.lower() == 'pria':
        kalori_basal = bbi * 30
    elif jenis_kelamin.lower() == 'wanita':
        kalori_basal = bbi * 25

    # Koreksi Usia
    if 60 <= usia <= 69:
        kalori_basal -= 0.1 * kalori_basal
    elif 40 <= usia <= 59:
        kalori_basal -= 0.05 * kalori_basal
    elif usia >= 70:
        kalori_basal -= 0.2 * kalori_basal
    
    # Koreksi Aktivitas
    if aktivitas.lower() == 'ringan':
        kalori_basal += kalori_basal * 0.15
    elif aktivitas.lower() == 'sedang':
        kalori_basal += kalori_basal * 0.25
    elif aktivitas.lower() == 'berat':
        kalori_basal += kalori_basal * 0.45
    
    # Menghitung kebutuhan nutrisi
    protein_kalori = 0.2 * kalori_basal
    lemak_kalori = 0.25 * kalori_basal
    karbohidrat_kalori = 0.5 * kalori_basal
    
    protein_gram = protein_kalori / 4
    lemak_gram = lemak_kalori / 9
    karbohidrat_gram = karbohidrat_kalori / 4
    
    return {
        "imt": imt,
        "bbi": bbi,
        "kalori_basal": kalori_basal,
        "protein_gram": protein_gram,
        "lemak_gram": lemak_gram,
        "karbohidrat_gram": karbohidrat_gram
    }
kebutuhan_gizi = hitung_akg_diabetes(berat_badan, tinggi, usia, jenis_kelamin)

# Membaca data makanan
df_filtered = pd.read_csv(resource_path('fix_dataset.csv'))

if filter == 4:
    df = df_filtered[df_filtered['Rating'] >= 4]
elif filter == 3:
    df = df_filtered[df_filtered['Rating'] >= 3]
elif filter == 2:
    df = df_filtered[df_filtered['Rating'] >= 2]
elif filter == 1:
    df = df_filtered[df_filtered['Rating'] >= 1]

# Membersihkan data
for col in ['Kalori', 'Karbohidrat', 'Protein', 'Lemak', 'Rating']:
    df[col] = df[col].astype(str).str.replace(',', '.').astype(float)

X = df[['Kalori', 'Karbohidrat', 'Protein', 'Lemak']].values
y = df['Rating'].values

# Mengelompokkan makanan menggunakan KMeans
kmeans = KMeans(n_clusters=5, random_state=42) 
df['Cluster'] = kmeans.fit_predict(X)

def calculate_nutrition_distance(features, target_features):
    return np.sqrt(np.sum((features - target_features) ** 2))

def fitness_function(combination, target_features, valid_X, cluster_labels, ratings):
    combined_features = np.sum(valid_X[list(combination)], axis=0)
    distance = calculate_nutrition_distance(combined_features, target_features)
    diversity_score = len(set(cluster_labels[list(combination)])) 
    average_rating = np.mean(ratings[list(combination)])  # Menghitung rata-rata rating dari kombinasi makanan
    return diversity_score * average_rating / (distance + 1e-6)

# Fungsi untuk merekomendasikan makanan menggunakan algoritma genetika
def recommend_meals_ga(target_features, valid_X, cluster_labels, ratings, population_size=100, num_generations=100):
    num_meals = len(valid_X)

    # Mengurutkan makanan berdasarkan rating dari yang tertinggi
    sorted_indices = np.argsort(ratings)[::-1]
    sorted_valid_X = valid_X[sorted_indices]
    sorted_cluster_labels = cluster_labels[sorted_indices]
    sorted_ratings = ratings[sorted_indices]

    population = [random.sample(range(num_meals), 3) for _ in range(population_size)]

    for generation in range(num_generations):
        fitness_scores = [fitness_function(individual, target_features, sorted_valid_X, sorted_cluster_labels, sorted_ratings) for individual in population]

        best_individual = population[np.argmax(fitness_scores)]
        best_fitness = max(fitness_scores)

        if best_fitness > 0.99:
            break

        new_population = []
        for _ in range(population_size):
            if random.random() < 0.2:
                new_individual = list(random.sample(range(num_meals), 3))
            else:
                parent1, parent2 = random.sample(population, 2)
                crossover_point = random.randint(0, 2)
                new_individual = parent1[:crossover_point] + parent2[crossover_point:]
            new_population.append(new_individual)
        population = new_population

    best_combination = best_individual
    recommended_meals = sorted_valid_X[list(best_combination)]
    recommended_ratings = sorted_ratings[list(best_combination)]
    return best_combination, recommended_meals, recommended_ratings

# Mengambil rating dari data yang valid
target_features = np.array([kebutuhan_gizi["kalori_basal"], kebutuhan_gizi["karbohidrat_gram"], kebutuhan_gizi["protein_gram"], kebutuhan_gizi["lemak_gram"]])

valid_indices = np.where(np.any(X != 0, axis=1))[0] 
valid_data = df.iloc[valid_indices]
valid_X = valid_data[['Kalori', 'Karbohidrat', 'Protein', 'Lemak']].values
ratings = valid_data['Rating'].values
cluster_labels = valid_data['Cluster'].values

best_combination, recommended_meals, recommended_ratings = recommend_meals_ga(target_features, valid_X, cluster_labels, ratings)

# Menghitung total kalori, karbohidrat, protein, dan lemak dari makanan yang direkomendasikan
kalori_basal = np.sum(recommended_meals[:, 0])
total_karbohidrat = np.sum(recommended_meals[:, 1])
total_protein = np.sum(recommended_meals[:, 2])
total_lemak = np.sum(recommended_meals[:, 3])

# Mencetak hasil
print("Total Kalori:", kebutuhan_gizi["kalori_basal"], "kalori per hari")
print("Kebutuhan Karbohidrat:", kebutuhan_gizi["karbohidrat_gram"], "gram per hari")
print("Kebutuhan Protein:", kebutuhan_gizi["protein_gram"], "gram per hari")
print("Kebutuhan Lemak:", kebutuhan_gizi["lemak_gram"], "gram per hari \n")
print("Rekomendasi makanan dan kandungan gizinya:")

for idx in best_combination:
    meal = valid_data.iloc[idx]
    print(f"{meal['Nama']} - Kalori: {meal['Kalori']}, Karbohidrat: {meal['Karbohidrat']}, Protein: {meal['Protein']}, Lemak: {meal['Lemak']}, Rating: {meal['Rating']}")

print("\nTotal Kalori dari makanan yang direkomendasikan:", kalori_basal)
print("Total Karbohidrat dari makanan yang direkomendasikan:", total_karbohidrat)
print("Total Protein dari makanan yang direkomendasikan:", total_protein)
print("Total Lemak dari makanan yang direkomendasikan:", total_lemak)
