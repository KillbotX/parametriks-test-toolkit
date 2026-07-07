import math
import numpy as np
import random
import statistics
from decimal import Decimal
from typing import Dict
import statsmodels.api as sm
import xgboost as xgb


class ParametriksPricingAgent:
    """
    Agent hybride combinant l'extraction sémantique multi-tâches (type BERT)
    et la validation actuarielle (Monte Carlo) pour identifier le surpricing.
    """
    def __init__(self, simulation_service: 'ScenarioSimulationService'):
        self.simulator = simulation_service
        self.min_combined_ratio_target = 0.85  
        
        self.appetite_labels = ["accept", "refer", "decline"]
        self.pricing_labels = ["low", "medium", "high"]

    def analyze_contract_and_price(
        self, 
        contract_text: str, 
        market_premium: Decimal, 
        historical_loss_ratio: float,
        exposure_count: int
    ) -> Dict:
        """
        Analyse un contrat pour détecter un décalage (mismatch) entre le risque réel
        et le prix du marché, puis quantifie l'opportunité face aux contraintes de solvabilité.
        """
        print(f"\n[Analyse Agent] Nouveau contrat soumis. Prime Marché: {market_premium} €")
        print(contract_text)
        bert_outputs = self._mock_multitask_inference(contract_text)
        
        pred_appetite = bert_outputs["appetite"]
        pred_pricing_band = bert_outputs["pricing_signal"]
        detected_clauses = bert_outputs["clauses"]
        
        print(f"  -> Clauses clés détectées: {', '.join(detected_clauses)}")
        print(f"  -> Tête B (Appétence Risque) : {pred_appetite.upper()}")
        print(f"  -> Tête C (Signal de Tarif)   : {pred_pricing_band.upper()} BAND")

        is_surpriced = False
        mismatch_reason = ""
        
        if pred_appetite == "accept" and pred_pricing_band == "high":
            is_surpriced = True
            mismatch_reason = "Mismatch Détecté: Risque technique FAIBLE (Accept) mais Tarification Marché ÉLEVÉE (High Band)."
        elif "limitation_of_liability" in detected_clauses and pred_pricing_band != "low":
            is_surpriced = True
            mismatch_reason = "Surpricing Structurel: Clause de limitation de responsabilité restrictive non intégrée dans le prix du marché."

        cap_per_event = None
        if "limitation_of_liability" in detected_clauses:
            cap_per_event = 50000.0  
            print(f"  -> Paramètre de production : Limitation de responsabilité détectée à {cap_per_event} €.")

        mc_results = self.simulator.run_monte_carlo(
            premium=market_premium,
            loss_ratio=historical_loss_ratio,
            exposure_count=exposure_count,
            iterations=10000,
            use_lognormal=True
        )

        expected_loss = mc_results["expected_loss"]
        var_99 = mc_results["var_99"]
        mean_loss = mc_results["mean"]
        
        technical_margin = float(market_premium) - mean_loss
        projected_combined_ratio = mean_loss / float(market_premium)
        var_99_coverage_ratio = float(market_premium) / var_99 if var_99 > 0 else 0

        decision = "REFER"
        action_plan = "Analyse humaine requise."
        proposed_premium = float(market_premium)

        if is_surpriced and projected_combined_ratio < self.min_combined_ratio_target:
            
            if var_99_coverage_ratio < 1.0:
                decision = "REFER_TO_HUMAN"
                proposed_premium = float(market_premium) 
                action_plan = (
                    f"OPPORTUNITÉ RENTABLE MAIS DANGEREUSE. Le profit moyen est validé (CR: {projected_combined_ratio*100:.1f}%), "
                    f"mais la VaR 99 ({var_99:.2f} €) excède la prime. "
                    f"Aiguillage obligatoire : Acheter un traité de réassurance Stop-Loss avant souscription."
                )
            else:
                decision = "SUBSCRIBE_IMMEDIATELY"
                proposed_premium = float(market_premium) * 0.95
                action_plan = (
                    f"Opportunité validée sans réserve. Proposer une contre-offre à {proposed_premium:.2f} € (-5%). "
                    f"Le capital couvre le scénario catastrophe (Ratio VaR: {var_99_coverage_ratio:.2f}x)."
                )
                
        elif pred_appetite == "decline":
            decision = "DECLINE"
            action_plan = "Risque hors appétence textuelle. Ne pas souscrire."

        return {
            "analysis": {
                "is_surpriced": is_surpriced,
                "reason": mismatch_reason,
                "projected_combined_ratio": projected_combined_ratio,
                "technical_margin_euros": technical_margin,
                "var_99_safety_multiplier": var_99_coverage_ratio
            },
            "actionable_decision": {
                "status": decision,
                "proposed_premium": proposed_premium,
                "recommendation": action_plan
            },
            "stochastic_metrics": mc_results
        }

    def _mock_multitask_inference(self, text: str) -> Dict:
        """
        Simule la sortie des couches d'activation d'un réseau BERT multi-tâches.
        """
        text_lower = text.lower()
        detected_clauses = []
        
        if "limitation" in text_lower or "responsabilité" in text_lower:
            detected_clauses.append("limitation_of_liability")
        if "arbitrage" in text_lower:
            detected_clauses.append("arbitration")
            
        if "flotte" in text_lower and "premium" in text_lower:
            return {
                "clauses": detected_clauses if detected_clauses else ["general_provisions"],
                "appetite": "accept",
                "pricing_signal": "high"
            }
        
        return {
            "clauses": ["general_provisions"],
            "appetite": "refer",
            "pricing_signal": "medium"
        }


class ScenarioSimulationService:
    """Moteur de modélisation des risques financiers via simulations stochastiques."""
    
    def run_monte_carlo(
        self,
        premium: Decimal,
        loss_ratio: float,
        exposure_count: int,
        iterations: int = 10000,
        use_lognormal: bool = True
    ) -> Dict:
        losses = []
        premium_f = float(premium)
        
        # Modélisation Log-normale pour capturer la "queue lourde" (sinistres rares mais graves)
        if use_lognormal and loss_ratio > 0:
            # Volatilité structurelle de la sinistralité (30% d'écart-type)
            sigma = 0.3  
            mu = math.log(loss_ratio) - 0.5 * (sigma ** 2)
        
        for _ in range(iterations):
            if use_lognormal and loss_ratio > 0:
                simulated_loss_ratio = random.lognormvariate(mu, sigma)
            else:
                 simulated_loss_ratio = max(0, random.gauss(loss_ratio, loss_ratio * 0.3))                
            loss_amount = premium_f * simulated_loss_ratio
            losses.append(loss_amount)

        losses.sort()
        
        def get_percentile(p: float) -> float:
            idx = min(int(iterations * p), iterations - 1)
            return losses[idx]

        return {
            "premium": premium_f,
            "expected_loss": premium_f * loss_ratio,
            "mean": sum(losses) / len(losses),
            "std_dev": statistics.stdev(losses) if len(losses) > 1 else 0,
            "var_95": get_percentile(0.95),
            "var_99": get_percentile(0.99),
            "percentiles": {
                "50": get_percentile(0.50),
                "75": get_percentile(0.75),
                "90": get_percentile(0.90),
                "95": get_percentile(0.95),
                "99": get_percentile(0.99),
            },
        }


class ModelComparisonService:
    """Service de comparaison de modèles actuariels pour comparer les primes simulées via InstantRisk."""
    def __init__(self, simulation_service: 'ScenarioSimulationService'):
        self.simulator = simulation_service
    
    def run_advanced_benchmarks(self, BASE_PREMIUM: float, base_loss_ratio: float, n_insureds: int = 200) -> Dict:
        """
        Génère un portefeuille d'assurés via Monte Carlo, puis entraîne 3 architectures :
          - Benchmark 1 : GLM Tweedie 
          - Benchmark 2 : Modèle Découplé Fréquence (Poisson) / Sévérité (Gamma)
          - Benchmark 3 : XGBoost avec Contraintes Monotones Basé Sur Tweedie
          - Benchmark 4 : XGBoost Quantreg avec un quantile référence à cibler (MAGIC_NUMBER)
        """
        X_risk_factor = np.random.uniform(1.0, 10.0, size=n_insureds)
        X_w_intercept = sm.add_constant(X_risk_factor)       
        Y_pure_premium = np.zeros(n_insureds)
        real_freq = np.zeros(n_insureds)  # Nombre de sinistres par assuré
        # paramètres par défaut - quantile par rapport auquel se fait le quantreg (similaire à celui observé avec 
        # les données); contraintes monotones pour les entraîneurs xgboost, sert à mieux entraîner cf 
        # https://xgboost.readthedocs.io/en/latest/tutorials/monotonic.html
        MAGIC_NUMBER = 0.89
        XGB_MONOTONE_CONSTRAINTS = (1,) 

        print(f"\n[Moteur MC] Génération de la sinistralité pour {n_insureds} assurés...")
        
        for i in range(n_insureds):
            individual_loss_ratio = base_loss_ratio * (X_risk_factor[i] / 5.0)
            mc_result = self.simulator.run_monte_carlo(
                premium=Decimal(str(BASE_PREMIUM)),
                loss_ratio=individual_loss_ratio,
                exposure_count=1,
                iterations=100,
                use_lognormal=True
            )
            
            Y_pure_premium[i] = mc_result["mean"]           
            lambda_freq = 0.2 * (X_risk_factor[i] / 5.0)  # ~0.2 sinistre par an en moyenne
            real_freq[i] = np.random.poisson(lambda_freq)


        print("--- Benchmark 1 : Ajustement du GLM Tweedie ---")
        model_tweedie = sm.GLM(
            Y_pure_premium, 
            X_w_intercept, 
            family=sm.families.Tweedie(link=sm.families.links.Log(), var_power=1.5)
        )
        res_tweedie = model_tweedie.fit()
        predictions_tweedie = res_tweedie.predict(X_w_intercept)
    

        print("--- Benchmark 2 : Ajustement du Modèle Fréquence/Sévérité ---")
        model_freq = sm.GLM(real_freq, X_w_intercept, family=sm.families.Poisson(link=sm.families.links.Log()))
        res_freq = model_freq.fit()
        
        claim_ind = real_freq > 0
        Y_non_null_claims = Y_pure_premium[claim_ind]
        X_non_null_claims = X_w_intercept[claim_ind]
        
        if len(Y_non_null_claims) > 0:
            model_sev = sm.GLM(
                Y_non_null_claims, 
                X_non_null_claims, 
                family=sm.families.Gamma(link=sm.families.links.Log())
            )
            res_sev = model_sev.fit()
            pred_sev = res_sev.predict(X_w_intercept)
        else:
            pred_sev = np.full(n_insureds, np.mean(Y_pure_premium))

        pred_freq_sev = res_freq.predict(X_w_intercept)
        final_sev_prediction = pred_freq_sev * pred_sev


        print("--- Benchmark 3 : Entraînement XGBoost Monotone ---")
        dtrain = xgb.DMatrix(X_risk_factor.reshape(-1, 1), label=Y_pure_premium)
        params_xgb = {
            'objective': 'reg:tweedie',         # Objectif XGB-GLMTweedie usuel pour l'assurance
            'tweedie_variance_power': 1.5,
            'monotone_constraints': XGB_MONOTONE_CONSTRAINTS,
            'learning_rate': 0.1,
            'max_depth': 4,
            'verbosity': 0
        }

        best_model = xgb.train(params_xgb, dtrain, num_boost_round=50)
        predictions_xgb = best_model.predict(dtrain)

        print("--- Benchmark 4 : Entraînement QuantReg XGBoost ---")        
        dtrain = xgb.DMatrix(X_risk_factor.reshape(-1, 1), label=Y_pure_premium)

        params_clone = {
            'objective': 'reg:quantileerror',     # Objectif de régression par quantile
            'quantile_alpha': MAGIC_NUMBER,               
            'max_depth': 6,                   
            'learning_rate': 0.1,             
            'verbosity': 0
        }
        model_clone = xgb.train(params_clone, dtrain, num_boost_round=100)
        predictions_clone = model_clone.predict(dtrain)
        avg_clone_premium = np.mean(predictions_clone)


        sample_comparisons = []
        for i in range(min(5, n_insureds)):
            sample_comparisons.append({
                "insured_id": i,
                "risk_score": round(X_risk_factor[i], 2),
                "real_mc_score": round(Y_pure_premium[i], 2),
                "tweedie_price": round(predictions_tweedie[i], 2),
                "freq_sev_price": round(final_sev_prediction[i], 2),
                "xgboost_price": round(float(predictions_xgb[i]), 2)
            })

        return {
            "totals": {
                "real_mc_loss": round(float(np.sum(Y_pure_premium)), 2),
                "global_tweedie_premium": round(float(np.sum(predictions_tweedie)), 2),
                "global_freq_sev_premium": round(float(np.sum(final_sev_prediction)), 2),
                "global_xgb_premium": round(float(np.sum(predictions_xgb)), 2),
                "global_quantreg_premium": round(float(avg_clone_premium), 2)
            },
            "sample_details": sample_comparisons
        }
    



# ----> BACKTEST SI STREAMLIT NE FONCTIONNE PAS


if __name__ == "__main__":
    AI_LOSS_RATIO = 0.65 
    BASE_PREMIUM = 150000.00
    N_INSUREDS = 45 
    CLAIM_TEXT = """
    Contrat de souscription - Assurance Flotte Automobile Corporative.
    Le souscripteur accepte les conditions de Tarification Premium du marché de Londres.
    CLAUSE DE SECURITE : Responsabilité civile de l'assureur limitée à un plafond strict 
    de 50 000 € par événement (Limitation de responsabilité mutuelle).
    """

    mc_engine = ScenarioSimulationService()   
    agent = ParametriksPricingAgent(simulation_service=mc_engine)
    comparer = ModelComparisonService(simulation_service=mc_engine)

    print(f"\n=== LANCEMENT DE L'ANALYSE COMPAREE (Portefeuille de {N_INSUREDS} assurés) ===")
    
    results = comparer.run_advanced_benchmarks(
        BASE_PREMIUM=BASE_PREMIUM,
        base_loss_ratio=AI_LOSS_RATIO,
        n_insureds=N_INSUREDS
    )

    poc_output = agent.analyze_contract_and_price(
        contract_text=CLAIM_TEXT,
        market_premium=Decimal("150000.00"),
        historical_loss_ratio=AI_LOSS_RATIO,
        exposure_count=N_INSUREDS
    )

    print("\n" + "="*60)
    print("       OUTPUT DU MODÈLE INITIAL (AGENT)   ")
    print("="*60)

    analysis = poc_output["analysis"]
    decision = poc_output["actionable_decision"]
    metrics = poc_output["stochastic_metrics"]

    print(f"Statut du Surpricing : {'DÉTECTÉ' if analysis['is_surpriced'] else 'NON DÉTECTÉ'}")
    print(f"Marge Bénéficiaire   : {analysis['technical_margin_euros']:.2f} € (Moyenne simulée)")
    print(f"Combined Ratio Estimé: {analysis['projected_combined_ratio']*100:.1f} %")
    print(f"Multiplicateur de Sécurité (Prime / VaR 99): {analysis['var_99_safety_multiplier']:.2f}x")
    print("-" * 60)
    print("DISTRIBUTION STOCHASTIQUE DES PERTES :")
    print(f"  - Perte Médiane (P50) : {metrics['percentiles']['50']:.2f} €")
    print(f"  - Perte à Risque (P95) : {metrics['percentiles']['95']:.2f} €")
    print(f"  - Scénario Catastrophe (VaR 99) : {metrics['percentiles']['99']:.2f} €")
    print("-" * 60)
    print("DÉCISION AUTOMATISÉE DE L'AGENT :")
    print(f"  Action Recommandée :  \033[92m{decision['status']}\033[0m")
    print(f"  Prix proposé ajusté:  {decision['proposed_premium']:.2f} €")
    print(f"  Plan d'action      :  {decision['recommendation']}")
    print("="*60)   
    print("\n" + "="*60)
    print("       COMPARAISON DES PRIMES GLOBALES DU PORTEFEUILLE")
    print("="*60)
    print(f"Charge Réelle Cumulée (Moteur MC) : {results['totals']['real_mc_loss'] / N_INSUREDS} €")
    print("-" * 60)
    print(f"Tarification GLM Tweedie          : {results['totals']['global_tweedie_premium']/ N_INSUREDS} €")
    print(f"Tarification Fréquence / Sévérité : {results['totals']['global_freq_sev_premium']/ N_INSUREDS} €")
    print(f"Tarification XGBoost Monotone     : {results['totals']['global_xgb_premium']/ N_INSUREDS}€")
    print(f"Tarification XGBoost QuantReg     : {results['totals']['global_quantreg_premium']}€")
    print("="*60)
    print("\nDÉTAIL COMPARATIF (ÉCHANTILLON D'ASSURÉS) :")
    print(f"{'ID':<5} | {'Score Risque':<12} | {'Coût MC':<10} | {'Tweedie':<10} | {'Freq/Sev':<10} | {'XGBoost':<10}")
    print("-" * 65)
    for row in results['sample_details']:
        print(f"#{row['insured_id']:<3} | "
              f"{row['risk_score']:<12.2f} | "
              f"{row['real_mc_score']:<10.2f} | "
              f"{row['tweedie_price']:<10.2f} | "
              f"{row['freq_sev_price']:<10.2f} | "
              f"{row['xgboost_price']:<10.2f}")
    print("-" * 65)