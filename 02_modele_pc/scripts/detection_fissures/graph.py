# generer_graphique_seuils.py
import matplotlib.pyplot as plt
import numpy as np

# Vos données réelles (depuis votre tableau)
seuils = [0.30, 0.35, 0.40, 0.45, 0.46, 0.48, 0.50, 0.55, 0.57, 0.60, 0.62, 0.65, 0.70]
precision = [87.07, 88.16, 89.29, 90.09, 90.91, 90.91, 90.87, 90.87, 91.20, 91.20, 91.63, 91.59, 92.38]
recall = [97.12, 96.63, 96.15, 96.15, 96.15, 96.15, 95.67, 95.67, 94.71, 94.71, 94.71, 94.23, 93.27]
f1 = [91.82, 92.20, 92.59, 93.02, 93.46, 93.46, 93.21, 93.21, 92.92, 92.92, 93.14, 92.89, 92.82]

plt.figure(figsize=(10, 6))
plt.plot(seuils, precision, 'b-o', label='Précision', linewidth=2)
plt.plot(seuils, recall, 'r-o', label='Recall', linewidth=2)
plt.plot(seuils, f1, 'g-o', label='F1-Score', linewidth=2)

plt.axvline(x=0.46, color='black', linestyle='--', alpha=0.7, label='Seuil optimal (0,46)')

plt.xlabel('Seuil de décision', fontsize=12)
plt.ylabel('Score (%)', fontsize=12)
plt.title('Évolution des performances en fonction du seuil de décision', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(80, 100)

plt.savefig('courbe_seuils_precise.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Graphique généré: courbe_seuils_precise.png")