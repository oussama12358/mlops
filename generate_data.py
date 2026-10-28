import csv
import numpy as np # numpy : pour générer des valeurs numériques aléatoires
# On fixe une graine aléatoire (seed). Cela ne demande pas à Python de
# commencer par le nombre 42 : ce nombre fixe simplement un point de #
# recréées à chaque exécution. C'est ce qui garantit la reproductibilité
# des expériences.
np.random.seed(42)
n = 500 # nombre de clients fictifs à générer
# On construit le tableau colonne par colonne.
# Chaque colonne ci-dessous est une FEATURE (donnée d'entrée du futur modèle).
ages = np.random.randint(18, 70, n) # âge entre 18 et 70 ans
salaires = np.random.randint(1200, 8000, n) # salaire mensuel simulé
abonnements = np.random.choice(["Basique", "Standard", "Premium"], n) # type d'offre
anciennetes = np.random.randint(0, 15, n) # ancienneté en années
consommations = np.random.randint(0, 100, n) # score d'usage du service, sur 100
# On crée maintenant la TARGET (colonne à prédire) : churn.
# Règle simplifiée et volontairement arbitraire : un client peu actif
# (consommation < 30) ET récent (ancienneté < 2 ans) est considéré
# comme « à risque de départ ».
churn = ((consommations < 30) & (anciennetes < 2)).astype(int)
data = [
	{
		"age": int(age),
		"salaire": int(salaire),
		"abonnement": str(abonnement),
		"anciennete": int(anciennete),
		"consommation": int(consommation),
		"churn": int(churn_value),
	}
	for age, salaire, abonnement, anciennete, consommation, churn_value in zip(
		ages, salaires, abonnements, anciennetes, consommations, churn
	)
]
# On sauvegarde le tableau dans un fichier CSV, sur le disque.
# ATTENTION : ce fichier n'est protégé par aucun système de version à ce stade.
with open("churn_data.csv", "w", newline="", encoding="utf-8") as csv_file:
	writer = csv.DictWriter(csv_file, fieldnames=data[0].keys())
	writer.writeheader()
	writer.writerows(data)
# Deux affichages de contrôle, juste pour vérifier visuellement le résultat.
print(data[:5])
print(f"Taux de churn : {churn.mean():.1%}")