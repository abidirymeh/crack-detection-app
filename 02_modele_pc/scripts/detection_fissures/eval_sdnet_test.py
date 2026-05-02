import tensorflow as tf
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

print("=" * 70)
print("📊 ÉVALUATION SUR SDNET TEST - TOUS LES SEUILS")
print("=" * 70)

# 1. Charger le modèle TFLite
interpreter = tf.lite.Interpreter('modele_enriched.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def predict_single(img):
    inp = np.expand_dims(img, 0).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], inp)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])[0][0]

# 2. Charger SDNET test
data = np.load('sdnet_prepared.npz')
X_test = data['X_test']
y_test = data['y_test']

print(f"\n📁 SDNET test: {len(X_test)} images")
print(f"   Saines: {sum(y_test == 0)}")
print(f"   Fissurées: {sum(y_test == 1)}")

# 3. Prédictions
print("\n🔄 Prédictions en cours...")
probas = []
for i, img in enumerate(X_test):
    probas.append(predict_single(img))
    if (i+1) % 100 == 0:
        print(f"   → {i+1}/{len(X_test)}")

probas = np.array(probas)

# 4. Tous les seuils de 0.30 à 0.80 par pas de 0.01
seuils = np.arange(0.30, 0.81, 0.01)

print("\n" + "=" * 70)
print("📊 RÉSULTATS SUR SDNET TEST - TOUS LES SEUILS")
print("=" * 70)

print(f"\n{'Seuil':>6} {'Précision':>10} {'Recall':>10} {'F1':>10} {'Accuracy':>10} {'VP':>6} {'FP':>6} {'VN':>6} {'FN':>6}")
print("-" * 75)

resultats = []

for seuil in seuils:
    y_pred = (probas > seuil).astype(int)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    vp = cm[1][1]  # Vrais positifs (fissures bien détectées)
    fp = cm[0][1]  # Faux positifs (saines classées fissures)
    vn = cm[0][0]  # Vrais négatifs (saines bien classées)
    fn = cm[1][0]  # Faux négatifs (fissures manquées)
    
    resultats.append((seuil, prec, rec, f1, acc, vp, fp, vn, fn))
    
    # Mettre en évidence le seuil optimal
    if 0.45 <= seuil <= 0.47:
        print(f"\033[92m{seuil:.2f}\033[0m   \033[92m{prec:>9.2%}\033[0m \033[92m{rec:>9.2%}\033[0m \033[92m{f1:>9.2%}\033[0m \033[92m{acc:>9.2%}\033[0m \033[92m{vp:>6}\033[0m \033[92m{fp:>6}\033[0m \033[92m{vn:>6}\033[0m \033[92m{fn:>6}\033[0m")
    elif seuil == 0.57:
        print(f"\033[94m{seuil:.2f}\033[0m   \033[94m{prec:>9.2%}\033[0m \033[94m{rec:>9.2%}\033[0m \033[94m{f1:>9.2%}\033[0m \033[94m{acc:>9.2%}\033[0m \033[94m{vp:>6}\033[0m \033[94m{fp:>6}\033[0m \033[94m{vn:>6}\033[0m \033[94m{fn:>6}\033[0m")
    else:
        print(f"{seuil:.2f}   {prec:>9.2%} {rec:>9.2%} {f1:>9.2%} {acc:>9.2%} {vp:>6} {fp:>6} {vn:>6} {fn:>6}")

# 5. Meilleurs seuils par objectif
print("\n" + "=" * 70)
print("🏆 MEILLEURS SEUILS PAR OBJECTIF")
print("=" * 70)

# Meilleur F1
best_f1 = max(resultats, key=lambda x: x[3])
print(f"\n🎯 MEILLEUR F1: seuil = {best_f1[0]:.2f}")
print(f"   → Précision: {best_f1[1]:.2%}, Recall: {best_f1[2]:.2%}, F1: {best_f1[3]:.2%}")

# Meilleure précision
best_prec = max(resultats, key=lambda x: x[1])
print(f"\n🎯 MEILLEURE PRÉCISION: seuil = {best_prec[0]:.2f}")
print(f"   → Précision: {best_prec[1]:.2%}, Recall: {best_prec[2]:.2%}, F1: {best_prec[3]:.2%}")

# Meilleur recall
best_rec = max(resultats, key=lambda x: x[2])
print(f"\n🎯 MEILLEUR RECALL: seuil = {best_rec[0]:.2f}")
print(f"   → Recall: {best_rec[2]:.2%}, Précision: {best_rec[1]:.2%}, F1: {best_rec[3]:.2%}")

# Meilleure accuracy
best_acc = max(resultats, key=lambda x: x[4])
print(f"\n🎯 MEILLEURE ACCURACY: seuil = {best_acc[0]:.2f}")
print(f"   → Accuracy: {best_acc[4]:.2%}, F1: {best_acc[3]:.2%}")

# 6. Générer un graphique
try:
    plt.figure(figsize=(12, 6))
    
    seuils_plot = [r[0] for r in resultats]
    prec_plot = [r[1] * 100 for r in resultats]
    rec_plot = [r[2] * 100 for r in resultats]
    f1_plot = [r[3] * 100 for r in resultats]
    acc_plot = [r[4] * 100 for r in resultats]
    
    plt.plot(seuils_plot, prec_plot, 'b-o', label='Précision', linewidth=2, markersize=4)
    plt.plot(seuils_plot, rec_plot, 'r-o', label='Recall', linewidth=2, markersize=4)
    plt.plot(seuils_plot, f1_plot, 'g-o', label='F1-Score', linewidth=2, markersize=4)
    plt.plot(seuils_plot, acc_plot, 'm-o', label='Accuracy', linewidth=2, markersize=4)
    
    plt.axvline(x=0.46, color='gold', linestyle='--', label='Meilleur F1 (0.46)', alpha=0.7, linewidth=2)
    plt.axvline(x=0.57, color='black', linestyle='--', label='Seuil Android (0.57)', alpha=0.7, linewidth=2)
    
    plt.xlabel('Seuil de décision', fontsize=12)
    plt.ylabel('Score (%)', fontsize=12)
    plt.title('Performance du modèle sur SDNET test - Tous les seuils', fontsize=14)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig('performance_sdnet_tous_seuils.png', dpi=150)
    print("\n✅ Graphique sauvegardé: performance_sdnet_tous_seuils.png")
except Exception as e:
    print(f"\n⚠️ Graphique non généré: {e}")

print("\n✅ Test terminé !")