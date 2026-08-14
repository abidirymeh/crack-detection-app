# generer_graphique.py
import matplotlib.pyplot as plt
import numpy as np

# Créer un exemple de graphique (remplacez par vos vraies données)
np.random.seed(42)
y_test = np.random.uniform(10, 50, 200)
y_pred = y_test * 0.95 + np.random.normal(0, 2, 200)

plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.6, edgecolors='k', linewidth=0.5)
plt.plot([10, 50], [10, 50], 'r--', linewidth=2, label='Prédiction parfaite')
plt.xlabel('Valeurs réelles (MPa)')
plt.ylabel('Prédictions (MPa)')
plt.title(f'Prédictions vs valeurs réelles\nR² = 0.93')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('resultats_materiaux.png', dpi=150)
print("✅ Graphique généré: resultats_materiaux.png")