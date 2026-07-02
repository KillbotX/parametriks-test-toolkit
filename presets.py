import numpy as np
import pandas as pd

def simulate_claims(years=5, seed=2026):
    """
    Simulates a claims history for 5 actuarial risk profiles.
    """
    np.random.seed(seed)
    claim_data = []
    
    INSURANCE_PROFILES = {
        "Auto": {
            "lambda": 120, 
            "dist": "gamma", 
            "param": {"shape": 2, "scale": 750} # Average = 1500€
        },
        "Industry": {
            "lambda": 15, 
            "dist": "lognormal", 
            "param": {"meanlog": 10.5, "sdlog": 1.2} # Median ~36k€
        },
        "Cyber": {
            "lambda": 4, 
            "dist": "pareto", 
            "param": {"alpha": 1.2, "xmin": 50000} # Very heavy tail
        },
        "Tech": {
            "lambda": 25, 
            "dist": "lognormal", 
            "param": {"meanlog": 9.2, "sdlog": 0.8}
        },
        "RC_General": {
            "lambda": 8, 
            "dist": "pareto", 
            "param": {"alpha": 1.6, "xmin": 20000}
        }
    }
    
    for year in range(1, years + 1):
        for profile, config in INSURANCE_PROFILES.items():
            n_claims = np.random.poisson(config["lambda"])
            if n_claims == 0:
                continue
                
            if config["dist"] == "gamma":
                claim_amounts = np.random.gamma(shape=config["param"]["shape"], scale=config["param"]["scale"], size=n_claims)
            elif config["dist"] == "lognormal":
                claim_amounts = np.random.lognormal(mean=config["param"]["meanlog"], sigma=config["param"]["sdlog"], size=n_claims)
            elif config["dist"] == "pareto":
                u = np.random.uniform(0, 1, size=n_claims)
                claim_amounts = config["param"]["xmin"] * (1 - u) ** (-1 / config["param"]["alpha"])
            
            for i in range(n_claims):
                claim_data.append({
                    "Year": year,
                    "Profile": profile,
                    "Claim_ID": f"S_{year}_{profile[:3].upper()}_{i+1}",
                    "Amount": round(claim_amounts[i], 2)
                })
                
    return pd.DataFrame(claim_data)

# Generation on module loading
df_claims = simulate_claims(years=5)
total_costs_by_sector = df_claims.groupby("Profile")["Amount"].sum().reset_index()
total_costs_by_sector["Avg_Annual_Cost"] = total_costs_by_sector["Amount"] / 5

# Market benchmark SFCR ratios
loss_ratios_benchmark = {
    "Auto": {"Leaders": 0.7210, "Intermediate": 0.7408, "Small_Startups": 0.6900},
    "Industry": {"Leaders": 0.6427, "Intermediate": 0.6512, "Small_Startups": 0.6475},
    "Cyber": {"Leaders": 0.4734, "Intermediate": 0.4487, "Small_Startups": 0.4829},
    "Tech": {"Leaders": 0.5859, "Intermediate": 0.5846, "Small_Syndicates": 0.5077},
    "RC_Generale": {"Leaders": 0.6548, "Intermediate": 0.6420, "Small_Syndicates": 0.7109}
}

avg_market_loss_ratios = {k: (v["Leaders"] + v["Intermediate"] + list(v.values())[2]) / 3 for k, v in loss_ratios_benchmark.items()}
total_costs_by_sector["Avg_Losses"] = total_costs_by_sector["Profile"].map(avg_market_loss_ratios)
total_costs_by_sector["Theoretical_Market_Premium"] = total_costs_by_sector["Avg_Annual_Cost"] / total_costs_by_sector["Avg_Losses"]
AI_WEIGHTS = {"Auto": 1.15, "Cyber": 1.60, "Industry": 1.30, "RC_Generale": 1.40, "Tech": 1.25}
total_costs_by_sector["Parametriks_Premium"] = total_costs_by_sector["Theoretical_Market_Premium"] * total_costs_by_sector["Profile"].map(AI_WEIGHTS)

def get_preset_data(Profile: str) -> dict:
    """
    Extracts settings from a sector to seamlessly populate the UI.
    """
    row = total_costs_by_sector[total_costs_by_sector["Profile"] == Profile].iloc[0]
    return {
        "market_premium": float(row["Theoretical_Market_Premium"]),
        "loss_ratio": float(row["Avg_Losses"]),
        "target_premium": float(row["Parametriks_Premium"]),
        "exposure": 120 if Profile == "Auto" else (25 if Profile == "Tech" else 15) # Aligned to history exposure
    }