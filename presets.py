import numpy as np
import pandas as pd

def simulate_claims(years=5, seed=42):
    """
    Simule un historique de sinistres pour 5 catégories de risques actuariels.
    """
    np.random.seed(seed)
    donnees = []
    
    profils = {
        "Auto": {
            "lambda": 120, 
            "loi": "gamma", 
            "param": {"shape": 2, "scale": 750}  # Moyenne = 1500€
        },
        "Industrie": {
            "lambda": 15, 
            "loi": "lognormal", 
            "param": {"meanlog": 10.5, "sdlog": 1.2}  # Médiane ~36k€
        },
        "Cyber": {
            "lambda": 4, 
            "loi": "pareto", 
            "param": {"alpha": 1.2, "xmin": 50000}  # Queue très lourde
        },
        "Tech": {
            "lambda": 25, 
            "loi": "lognormal", 
            "param": {"meanlog": 9.2, "sdlog": 0.8}
        },
        "RC_Generale": {
            "lambda": 8, 
            "loi": "pareto", 
            "param": {"alpha": 1.6, "xmin": 20000}
        }
    }
    
    for annee in range(1, years + 1):
        for cat, config in profils.items():
            n_sinistres = np.random.poisson(config["lambda"])
            if n_sinistres == 0:
                continue
                
            if config["loi"] == "gamma":
                couts = np.random.gamma(shape=config["param"]["shape"], scale=config["param"]["scale"], size=n_sinistres)
            elif config["loi"] == "lognormal":
                couts = np.random.lognormal(mean=config["param"]["meanlog"], sigma=config["param"]["sdlog"], size=n_sinistres)
            elif config["loi"] == "pareto":
                u = np.random.uniform(0, 1, size=n_sinistres)
                couts = config["param"]["xmin"] * (1 - u) ** (-1 / config["param"]["alpha"])
            
            for i in range(n_sinistres):
                donnees.append({
                    "Annee": annee,
                    "Categorie": cat,
                    "ID_Sinistre": f"S_{annee}_{cat[:3].upper()}_{i+1}",
                    "Montant": round(couts[i], 2)
                })
                
    return pd.DataFrame(donnees)

# Génération au chargement du module
df_sinistres = simulate_claims(years=5)
calcul_secteurs = df_sinistres.groupby("Categorie")["Montant"].sum().reset_index()
calcul_secteurs["Cout_Annuel_Moyen"] = calcul_secteurs["Montant"] / 5

# Ratios SFCR benchmark du marché
loss_ratios_benchmark = {
    "Auto": {"Leaders": 0.7210, "Intermédiaires": 0.7408, "Petites_InsurTechs": 0.6900},
    "Industrie": {"Leaders": 0.6427, "Intermédiaires": 0.6512, "Petites_InsurTechs": 0.6475},
    "Cyber": {"Leaders": 0.4734, "Intermédiaires": 0.4487, "Petites_InsurTechs": 0.4829},
    "Tech": {"Leaders": 0.5859, "Intermédiaires": 0.5846, "Petites_Mutuelles": 0.5077},
    "RC_Generale": {"Leaders": 0.6548, "Intermédiaires": 0.6420, "Petites_Mutuelles": 0.7109}
}

ratios_moyens_marche = {k: (v["Leaders"] + v["Intermédiaires"] + list(v.values())[2]) / 3 for k, v in loss_ratios_benchmark.items()}
calcul_secteurs["Ratio_SFCR_Moyen"] = calcul_secteurs["Categorie"].map(ratios_moyens_marche)
calcul_secteurs["Prime_Marche_Traditionnel"] = calcul_secteurs["Cout_Annuel_Moyen"] / calcul_secteurs["Ratio_SFCR_Moyen"]
surcharges_ia = {"Auto": 1.15, "Cyber": 1.60, "Industrie": 1.30, "RC_Generale": 1.40, "Tech": 1.25}
calcul_secteurs["Prime_Parametriks"] = calcul_secteurs["Prime_Marche_Traditionnel"] * calcul_secteurs["Categorie"].map(surcharges_ia)

def get_preset_data(categorie: str) -> dict:
    """
    Extrait les paramètres d'un secteur pour alimenter l'UI de manière transparente.
    """
    row = calcul_secteurs[calcul_secteurs["Categorie"] == categorie].iloc[0]
    return {
        "market_premium": float(row["Prime_Marche_Traditionnel"]),
        "loss_ratio": float(row["Ratio_SFCR_Moyen"]),
        "target_premium": float(row["Prime_Parametriks"]),
        "exposure": 120 if categorie == "Auto" else (25 if categorie == "Tech" else 15) # Aligné sur l'exposition de l'historique
    }