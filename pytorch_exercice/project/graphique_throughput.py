import csv
import numpy as np
import matplotlib.pyplot as plt


batch_size = 32

# Fonction pour lire les données CPU / Temps d'exécution
def read_cpu_execution_data(file_path):
    cpus, loading_times, computation_times = [], [], []
    
    # Dictionnaire pour stocker le maximum des valeurs par nombre de CPU
    max_loading_times = {}
    max_computation_times = {}

    with open(file_path, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Ignorer l'en-tête
        
        for row in reader:
            num_cpus = int(row[0])  # Nombre de CPU
            loading_time = float(row[1])  # Temps de chargement
            computation_time = float(row[2])  # Temps de calcul + communication
            
            # Mettre à jour le maximum pour le temps de chargement
            if num_cpus not in max_loading_times:
                max_loading_times[num_cpus] = loading_time
            else:
                max_loading_times[num_cpus] = max(max_loading_times[num_cpus], loading_time)
            
            # Mettre à jour le maximum pour le temps de calcul + communication
            if num_cpus not in max_computation_times:
                max_computation_times[num_cpus] = computation_time
            else:
                max_computation_times[num_cpus] = max(max_computation_times[num_cpus], computation_time)

    # Afficher les résultats
    for num_cpus in sorted(max_loading_times.keys()):
        print(f"Nombre de CPU: {num_cpus}")
        print(f"  Temps de chargement maximum: {max_loading_times[num_cpus]}")
        print(f"  Temps de calcul + communication maximum: {max_computation_times[num_cpus]}")
        
        cpus.append(num_cpus)  # Nombre de CPU
        loading_times.append(max_loading_times[num_cpus])  # Temps de chargement
        computation_times.append(max_computation_times[num_cpus])  # Temps de calcul + communication

    return cpus, loading_times, computation_times


# Fonction pour générer le graphique en barres empilées
def plot_throughput(file_path):
    cpus, loading_times, computation_times = read_cpu_execution_data(file_path)

    # Convertir les listes en tableaux NumPy pour plus de flexibilité
    cpus = np.array(cpus)
    loading_times = np.array(loading_times)
    computation_times = np.array(computation_times)
    
    # Hauteur totale de chaque barre (somme des deux temps)
    total_times = loading_times + computation_times
    throughput= batch_size/total_times

    # Création du graphique
    plt.plot(cpus, throughput)

    # Ajout des annotations
    for i in range(len(cpus)):
        plt.text(cpus[i], throughput[i] + 0.2, f"{throughput[i]:.2f}s", ha='center', fontsize=10, fontweight='bold')

    # Configuration du graphique
    plt.xlabel("Nombre de CPU")
    plt.ylabel("Débit (secondes)")
    plt.title("Impact du nombre de CPU sur le débit")
    plt.xticks(cpus)  # Afficher uniquement les valeurs de CPU
    
    # Afficher le graphique
    plt.show()

# Exécuter la fonction avec le fichier CSV

plot_throughput('training_time.csv')
plot_throughput('training_time_gpu.csv')