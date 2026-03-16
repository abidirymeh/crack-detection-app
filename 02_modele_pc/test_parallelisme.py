import tensorflow as tf
import time
import numpy as np
import matplotlib.pyplot as plt

print("🔬 TEST DU PARALLÉLISME")
print("=" * 40)

# Charger MobileNetV3-Small (notre champion)
model = tf.keras.applications.MobileNetV3Small(
    input_shape=(224, 224, 3),
    weights='imagenet'
)

dummy_input = np.random.rand(1, 224, 224, 3).astype(np.float32)

# Tester différents nombres de threads
threads_configs = [1, 2, 4, 6, 8]
latences = []

for n_threads in threads_configs:
    print(f"⏳ Test avec {n_threads} thread(s)...")
    
    # Configurer le nombre de threads
    tf.config.threading.set_intra_op_parallelism_threads(n_threads)
    tf.config.threading.set_inter_op_parallelism_threads(n_threads)
    
    # Warm-up
    for _ in range(10):
        model.predict(dummy_input, verbose=0)
    
    # Mesure
    latence_moy = []
    for _ in range(30):
        start = time.time()
        model.predict(dummy_input, verbose=0)
        end = time.time()
        latence_moy.append((end - start) * 1000)
    
    latences.append(np.mean(latence_moy))
    print(f"   → {latences[-1]:.2f} ms")

# Afficher le graphique
plt.figure(figsize=(10, 6))
plt.plot(threads_configs, latences, 'bo-', linewidth=2, markersize=8)
plt.xlabel('Nombre de threads', fontsize=12)
plt.ylabel('Latence (ms)', fontsize=12)
plt.title('Impact du parallélisme sur la latence - MobileNetV3-Small', fontsize=14)
plt.grid(True, alpha=0.3)
plt.xticks(threads_configs)

# Calculer le gain
gain = latences[0] / latences[-1]
plt.text(0.5, 0.95, f'Gain entre 1 et {threads_configs[-1]} threads: x{gain:.2f}', 
         transform=plt.gca().transAxes, fontsize=11)

plt.savefig('parallelisme.png', dpi=150)
plt.show()

print(f"\n✅ Test terminé !")
print(f"📈 Gain entre 1 et {threads_configs[-1]} threads: x{gain:.2f}")
print(f"📊 Graphique sauvegardé: parallelisme.png")