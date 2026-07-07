import streamlit as st
from decimal import Decimal
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Importation du backend, presets, base de ml
from backend import (
    ScenarioSimulationService, 
    ParametriksPricingAgent, 
    ModelComparisonService
)
import presets
from geneticalg import EvolutionaryPricingOptimizer




#PARAMÈTRES PAR DÉFAUT
DEFAULT_PREMIUM = 150000.0
DEFAULT_LR = 65
DEFAULT_EXPOSURE = 45
SAMPLE_TEXTS = {
    "Saisie Manuelle": "Contrat standard - Flotte Automobile Corporative.\nCLAUSE DE SECURITE : Responsabilité limitée à un plafond strict de 50 000 €.",
    "Auto": "Contrat de Flotte Automobile - 120 véhicules légers.\nCLAUSE DE SECURITE : Limitation de responsabilité à 50 000 € par accrochage.",
    "Industrie": "Risque Dommage aux Biens & Pertes d'Exploitation Industrielles.",
    "Cyber": "Police d'Assurance Cyber Risques - Infrastructures Critiques.\nCLAUSE DE SECURITE : Cap global des pertes à 50 000 € par rançongiciel.",
    "Tech": "Responsabilité Civile Professionnelle Erreurs & Omissions (E&O).",
    "RC_Generale": "Contrat Responsabilité Civile Générale Corporative."
}


# streamlit config
st.set_page_config(
    page_title="Parametriks Pricing Agent",
    page_icon="🦅",
    layout="wide"
)
st.markdown("""
    <style>
    /* =========================================================================
       1. FONDS ET CONFIGURATION STRUCTURELLE (DARK MODE)
       ========================================================================= */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    .stApp {
        background-color: #0B0F19; /* Fond de page : Bleu nuit de l'espace profond */
        font-family: 'Inter', -apple-system, sans-serif;
        color: #E2E8F0;
    }

    /* Titres Typographiques Épurés */
    h1, h2, h3, h4 {
        color: #F8FAFC !important; /* Blanc pur avec un léger reflet */
        font-weight: 600 !important;
        letter-spacing: -0.02em !important;
    }
    
    /* Nettoyage des bordures par défaut de Streamlit */
    hr {
        border-color: #1E293B !important;
    }

    header, footer {
        visibility: hidden !important;
    }

    /* =========================================================================
       2. CRÉATION DES CARTES VITRÉES INTERACTIVES (st.metric)
       ========================================================================= */
    div[data-testid="stMetric"] {
        background-color: #111827 !important; /* Gris anthracite de conteneur */
        border: 1px solid #1F2937 !important; /* Bordure subtile */
        padding: 20px 24px !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2), 0 2px 4px -2px rgba(0, 0, 0, 0.2) !important;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    }
    div[data-testid="stMetric"]:hover {
        border-color: #3b82f6 !important; /* Éclat bleu électrique discret au survol */
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.4) !important;
    }
    
    /* Customisation des textes à l'intérieur des métriques */
    div[data-testid="stMetricLabel"] p {
        color: #94A3B8 !important; /* Gris moyen pour le contraste secondaire */
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    div[data-testid="stMetricValue"] div {
        color: #F8FAFC !important;
        font-size: 1.8rem !important;
        font-weight: 700 !important;
    }

    /* =========================================================================
       3. BOUTON CYBER-ACTION (Call To Action)
       ========================================================================= */
    div.stButton > button:first-child {
        background-color: #3B82F6 !important; /* Bleu Électrique Vercel */
        color: #FFFFFF !important;
        border: 1px solid #2563EB !important;
        border-radius: 8px !important;
        padding: 14px 28px !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        width: 100%;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.2) !important;
        transition: all 0.2s ease !important;
    }
    div.stButton > button:first-child:hover {
        background-color: #2563EB !important;
        box-shadow: 0 12px 20px -3px rgba(59, 130, 246, 0.4) !important;
        transform: translateY(-1px);
    }

    /* =========================================================================
       4. CONTRÔLES FORMULAIRES & BARRE LATÉRALE CYBERPUNK CHIC
       ========================================================================= */
    section[data-testid="stSidebar"] {
        background-color: #0F172A !important; /* Ardoise très sombre */
        border-right: 1px solid #1E293B !important;
    }
    
    /* Inputs, sélecteurs et zones de texte unifiés */
    div[data-baseweb="select"], div[data-baseweb="input"], textarea {
        background-color: #0F172A !important;
        color: #F8FAFC !important;
        border-radius: 8px !important;
        border: 1px solid #1E293B !important;
    }
    
    /* Fix pour la couleur du texte saisi dans les inputs */
    input, textarea {
        color: #F8FAFC !important;
    }

    /* Accordéons (st.expander) au design intégré */
    div[data-testid="stExpander"] {
        background-color: #111827 !important;
        border: 1px solid #1F2937 !important;
        border-radius: 8px !important;
    }
    </style>
""", unsafe_allow_html=True)

st.title("Parametriks AI Pricing Cockpit")
st.subheader("Analyse du pricing des modèles de Parametriks et comparaison effective avec d'autres modèles")
st.markdown("---")


st.sidebar.header("Portée de la Simulation")
sim_code = st.sidebar.radio(
    "Choisir le niveau d'analyse :",
    options=["Analyse Unitaire (Détail Contrat)", "Multi-Simulations (Stress-Test Portefeuille)", "Entraînement modèle"]
)

st.sidebar.header("Chargement d'un Preset Portefeuille")

# Choix du type de portefeuille
preset_choice = st.sidebar.selectbox(
    "Sélectionner un profil de risque historique :",
    options=["Saisie Manuelle", "Auto", "Industrie", "Cyber", "Tech", "RC_Generale"]
)

if preset_choice != "Saisie Manuelle":
    preset_data = presets.get_preset_data(preset_choice)
    DEFAULT_PREMIUM = preset_data["market_premium"]
    DEFAULT_LR = int(preset_data["loss_ratio"] * 100)
    DEFAULT_EXPOSURE = preset_data["exposure"]
    st.sidebar.success(f"Données historiques '{preset_choice}' injectées !")

st.sidebar.header("Paramètres de Souscription")

market_premium_input = st.sidebar.number_input(
    "Prime Proposée par le Marché (€)", 
    min_value=1000.0, value=float(DEFAULT_PREMIUM), step=5000.0
)
historical_lr_input = st.sidebar.slider(
    "Loss Ratio Historique Sectoriel (%)", 
    min_value=10, max_value=150, value=int(DEFAULT_LR)
) / 100.0
exposure_input = st.sidebar.number_input(
    "Taille de la Flotte (Unités d'exposition)", 
    min_value=5, max_value=500, value=int(DEFAULT_EXPOSURE)
)
if sim_code == "Multi-Simulations (Stress-Test Portefeuille)":
    st.sidebar.markdown("---")
    st.sidebar.header("Paramètres")
    n_sim_input = st.sidebar.slider(
        "Nombre de simulations", 
        min_value=100, max_value=5000, value=1000, step=100
    )
    kpi_choisi = st.sidebar.selectbox(
        "KPI de comparaison de modèles :",
        options=["Sharpe Ratio", "Résilience aux Chocs de Queue (TVaR 95%)", "Efficacité du Capital Économique (RoRC)", "Indice de Prudence (Marge de Sécurité)"]
    )
elif sim_code == "Entraînement modèle":
    st.sidebar.markdown("---")
    st.sidebar.header("Paramètres de l'optimisation génétique")  
    total_generations = st.sidebar.slider("Générations Totales", 10, 100, 40, 10)
    population_size_input = st.sidebar.slider("Taille de la Population", 10, 50, 20, 5)
    mutation_rate_input = st.sidebar.slider("Taux de Mutation (%)", 5, 50, 15, 5) / 100.0  
    kpi_choisi = st.sidebar.selectbox(
        "Objectif d'Optimisation (KPI Target) :",
        options=["Ratio Risque/Rendement (Sharpe Adapté)", "Résilience aux Chocs de Queue (TVaR 95%)", "Efficacité du Capital Économique (RoRC)", "Indice de Prudence (Marge de Sécurité)"]
    )


st.header("Clauses Contractuelles Soumises")
contract_text_input = st.text_area(
    "Brut du traité soumis :", value=SAMPLE_TEXTS[preset_choice], height=100
)

label_bouton = "Lancer l'Analyse Unitaire" if "Unitaire" in sim_code else "Exécuter le Stress-Test (Multi-Simulations)"
analyze_button = st.button(label_bouton, use_container_width=True)



if analyze_button:
    # Initialisation des moteurs de calcul
    mc_engine = ScenarioSimulationService()
    agent = ParametriksPricingAgent(simulation_service=mc_engine)
    comparer = ModelComparisonService(simulation_service=mc_engine)
    if "Unitaire" in sim_code:
        print(contract_text_input)
        with st.spinner("Analyse sémantique (BERT), simulations Monte Carlo et ajustements GLM/XGBoost en cours..."):
            poc_output = agent.analyze_contract_and_price(
                contract_text=contract_text_input,
                market_premium=Decimal(str(market_premium_input)),
                historical_loss_ratio=historical_lr_input,
                exposure_count=exposure_input
            )
            adjusted_loss_ratio = historical_lr_input * 0.85 if "limitation" in contract_text_input.lower() else historical_lr_input
        
            bench_results = comparer.run_advanced_benchmarks(
                BASE_PREMIUM=market_premium_input,
                base_loss_ratio=adjusted_loss_ratio,
                n_insureds=exposure_input
            )
        analysis = poc_output["analysis"]
        decision = poc_output["actionable_decision"]
        metrics = poc_output["stochastic_metrics"]
        status = decision['status']

        st.header("1. Décision Automatisée & Raisonnement de l'Agent")
        if status == "SUBSCRIBE_IMMEDIATELY":
            st.success(f"### 🔥 ACTION DIRECTE : {status.replace('_', ' ')}")
        elif "REFER" in status:
            st.warning(f"### ⚠️ STATUT PRUDENTIEL : {status.replace('_', ' ')}")
        else:
            st.error(f"### ❌ REFUS SYSTÉMIQUE : {status.replace('_', ' ')}")
        
        with st.status("Cliquer pour déplier le cheminement de pensée de l'IA (Explainable AI)", expanded=True) as status_box:
            st.markdown(f"**1. Étape Sémantique (BERT) :** L'algorithme a scanné le document texte.")
            st.write(f"*Résultat :* {analysis['reason']}")
        
            st.markdown(f"**2. Étape Stochastique (Monte Carlo) :** 10 000 années de sinistralité ont été projetées.")
            st.write(f"*Résultat :* Le coût moyen simulé est de **{metrics['mean']:,.2f} €** face à une prime demandée de **{market_premium_input:,.2f} €**.")
        
            st.markdown(f"**3. Décision de l'agent:**")
            if status == "REFER_TO_HUMAN":
                st.write(f"🛑 *Pourquoi ce blocage ?* Bien que l'opportunité de profit soit réelle (Combined Ratio de {analysis['projected_combined_ratio']*100:.1f}%), la perte catastrophe à 99% (**{metrics['var_99']:,.2f} €**) dépasse vos capacités. L'agent refuse l'immédiateté commerciale pour **protéger vos fonds propres**.")
            elif status == "SUBSCRIBE_IMMEDIATELY":
                st.write(f"✅ *Pourquoi ce feu vert ?* Le deal est rentable (Combined Ratio de {analysis['projected_combined_ratio']*100:.1f}%) ET la prime couvre largement le scénario de ruine à 99% (Ratio de couverture : {analysis['var_99_safety_multiplier']:.2f}x).")
            else:
                st.write(f"Le profil de risque ne valide pas les critères de souscription automatiques.")
        
            status_box.update(label="Raisonnement de l'agent", state="complete")
        st.info(f"**📋 Plan d'action recommandé :** {decision['recommendation']}")


        st.markdown("#### Métriques d'Analyse")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Combined Ratio Projeté", f"{analysis['projected_combined_ratio']*100:.1f} %", help="Inférieur à 85% = Excellent")
        m2.metric("Prime proposée", f"{decision['proposed_premium']:.1f} €", help="Inférieur à 85% = Excellent")
        m3.metric("Marge Technique Attendue", f"{analysis['technical_margin_euros']:,.2f} €")
        m4.metric("Scénario Catastrophe (VaR 99)", f"{metrics['percentiles']['99']:,.2f} €")
        m5.metric("Couverture VaR 99", f"{analysis['var_99_safety_multiplier']:.2f}x", 
                    delta="Sécurisé" if analysis['var_99_safety_multiplier'] >= 1.0 else "Sous-capitalisé",
                    delta_color="normal" if analysis['var_99_safety_multiplier'] >= 1.0 else "inverse")

        st.markdown("---")
        st.markdown("#### Courbe de Risque & Positionnement de la Prime")

            # Recréation de la distribution cumulative (simulation des percentiles de P50 à P99.9)
        percentiles_axes = ['50', '75', '90', '95', '99']
        losses = [metrics['percentiles'][p] for p in percentiles_axes]
        fig_distribution = go.Figure()

        fig_distribution.add_trace(go.Scatter(
                x=percentiles_axes, 
                y=losses,
                mode='lines+markers',
                name='Pertes Cumulées (Monte Carlo)',
                line=dict(color='#2980b9', width=3),
                hovertemplate='Percentile %{x} : %{y:,.2f} €<extra></extra>'
            ))

        fig_distribution.add_trace(go.Scatter(
                x=percentiles_axes,
                y=[market_premium_input] * len(percentiles_axes),
                mode='lines',
                name='Prime Marché Proposée',
                line=dict(color='#e74c3c', width=2, dash='dash'),
                hovertemplate='Prime : %{y:,.2f} €<extra></extra>'
            ))

        fig_distribution.update_layout(
                title=f"Distribution de la Queue de Soumission ({preset_choice}) vs Prime",
                xaxis_title="Percentile de Sévérité (Scénarios du plus probable au plus extrême)",
                yaxis_title="Montant des Pertes / Primes (€)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=20, r=20, t=60, b=20),
                height=350
            )

        st.plotly_chart(fig_distribution, use_container_width=True)
        st.caption("**Interprétation visuelle :** Si la ligne rouge (Prime) passe en dessous de la courbe bleue avant le point P99, le risque de ruine statistique est matériel. C'est exactement ce repère visuel qui déclenche l'alerte de l'Agent.")
        st.header("2. Benchmark de Tarification Portefeuille")
        st.write("Visualisation de la capacité des différents modèles du marché à capter le surpricing sur un portefeuille global.")
        totals = bench_results["totals"] 
        col_graph, col_table = st.columns([1.2, 1.8])
    
        with col_graph:
        # Transformation pour graphique Streamlit
            df_chart = pd.DataFrame({
            "Prime Globale (€)": [
                totals["real_mc_loss"], 
                totals["global_tweedie_premium"], 
                totals["global_freq_sev_premium"], 
                totals["global_xgb_premium"]
            ]
            }, index=["Charge Réelle (MC)", "GLM Tweedie", "Fréquence / Sévérité", "XGBoost Monotone"])
        
            st.bar_chart(df_chart, use_container_width=True)

        with col_table:
            df_table = pd.DataFrame({
            "Architecture du Modèle": ["Charge Réelle (Moteur MC)", "GLM Tweedie (Standard)", "Découplé Fréquence/Sévérité", "XGBoost Monotone (ML Controlled)"],
            "Collecte Totale Portefeuille": [
                f"{totals['real_mc_loss']/DEFAULT_EXPOSURE} €", 
                f"{totals['global_tweedie_premium']/DEFAULT_EXPOSURE} €", 
                f"{totals['global_freq_sev_premium']/DEFAULT_EXPOSURE} €", 
                f"{totals['global_xgb_premium']/DEFAULT_EXPOSURE} €"
            ],
            "Statut vs Coût Réel": [
                "Baseline",
                "🔴 SURPRICED" if totals["global_tweedie_premium"] > totals["real_mc_loss"] else "🔵 UNDERPRICED",
                "🔴 SURPRICED" if totals["global_freq_sev_premium"] > totals["real_mc_loss"] else "🔵 UNDERPRICED",
                "🔴 SURPRICED" if totals["global_xgb_premium"] > totals["real_mc_loss"] else "🔵 UNDERPRICED"
            ]
            })
            st.dataframe(df_table, hide_index=True, use_container_width=True)
        
            st.markdown("**Échantillon Profilé (Zoom Assurés 0 à 4) :**")
            df_gran = pd.DataFrame(bench_results["sample_details"])
            df_gran.columns = ["ID", "Score Risque", "Coût MC (€)", "Tweedie (€)", "Freq/Sev (€)", "XGBoost (€)"]
            st.dataframe(df_gran.style.format(precision=2), hide_index=True, use_container_width=True)

            st.markdown("---")


            st.header("3. XGBoost quantile regression (P89)")
            st.write("Validation par reproduction de la courbe de tarification adverse via un calage de quantile de stress à 89%.")
    
            c_repro1, c_repro2, c_repro3 = st.columns(3)
    
            c_repro1.metric(
                label="Tarif Moyen du Clone Modélisé (P89)", 
                value=f"{bench_results['totals']['global_quantreg_premium']:,.2f} €"
            )
    
    
        st.caption("🎯 **Note technique :** Si l'écart est faible, cela démontre mathématiquement que la tarification du concurrent n'est pas linéaire mais suit précisément une politique de marge calée sur le 89ème percentile des pertes possibles.")

    elif "Multi-Simulations" in sim_code:
        with st.spinner(f"Génération de {n_sim_input} trajectoires de marché macro..."):
            # Simulation d'un vecteur de résultats macro sur N itérations
            mu_perte = market_premium_input * historical_lr_input
            sigma_perte = mu_perte * (0.3 if "limitation" in contract_text_input.lower() else 0.6)
            
            # Simulation de trajectoires (Lognormale pour imiter la sinistralité agrégée)
            shape = np.sqrt(np.log(1 + (sigma_perte / mu_perte) ** 2))
            scale = mu_perte / np.sqrt(1 + (sigma_perte / mu_perte) ** 2)
            trajectoires_pertes = np.random.lognormal(mean=np.log(scale), sigma=shape, size=n_sim_input)
            
            # Calcul des indicateurs macro de base
            primes_collectees_totatles = market_premium_input * n_sim_input
            pertes_totales_simulees = np.sum(trajectoires_pertes)
            combined_ratio_moyen_macro = pertes_totales_simulees / primes_collectees_totatles
            proba_ruine_macro = np.mean(trajectoires_pertes > market_premium_input)

            # Simulation des comportements de tarification globale des différents modèles
            charge_ia = presets.AI_WEIGHTS.get(preset_choice, 1.30)           
            prime_moy_parametriks = market_premium_input * charge_ia
            prime_moy_xgboost = market_premium_input * (1.18 if preset_choice in ["Auto", "Tech"] else 1.45)
            prime_moy_tweedie = market_premium_input * (1.05 if preset_choice == "Auto" else 1.12)
            prime_moy_freq_sev = market_premium_input * 1.20

        st.header("1. Tableau de Bord de Stress-Testing Macro (Portefeuille)")
        st.write(f"Analyse de stabilité stochastique sur la base de **{n_sim_input:,} trajectoires** indépendantes.")

        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Combined Ratio Moyen (Marché)", f"{combined_ratio_moyen_macro * 100:.2f} %")
        status_color = "inverse" if proba_ruine_macro > 0.15 else "normal"
        mc2.metric(
            "Probabilité de Ruine (Tarif Marché)", 
            f"{proba_ruine_macro * 100:.1f} %",
            delta="Seuil Critique (>15%)" if proba_ruine_macro > 0.15 else "Zone Sécurisée",
            delta_color=status_color
        )
        mc3.metric("Pire Scénario Généré (Max)", f"{np.max(trajectoires_pertes):,.2f} €")

        st.markdown("---")
        st.subheader("2. Benchmark de Pricing Macro & Sélection du Modèle Optimal")
        

        # Calcul préalable des quantiles des pertes
        p95_macro = np.percentile(trajectoires_pertes, 95)
        p99_macro = np.percentile(trajectoires_pertes, 99)
        tvar_95_macro = np.mean(trajectoires_pertes[trajectoires_pertes >= p95_macro])
        # calcul de la probabilité de ruine specifique à chaque modèle
        ruine_parametriks = np.mean(trajectoires_pertes > prime_moy_parametriks)
        ruine_xgboost = np.mean(trajectoires_pertes > prime_moy_xgboost)
        ruine_tweedie = np.mean(trajectoires_pertes > prime_moy_tweedie)
        ruine_freq_sev = np.mean(trajectoires_pertes > prime_moy_freq_sev)


        #Sharpe ratio, Rorc, TVaR...
        def calculate_model_score(prime_mod, proba_ruine, nom_kpi):
            marge_brute = (prime_mod - mu_perte) / prime_mod          
            if nom_kpi == "Sharpe Ratio":
                return max(0.0, marge_brute - (proba_ruine * 2.5))
                
            elif nom_kpi == "TVaR 95%":
                return float(min(100.0, (prime_mod / tvar_95_macro) * 100))
                
            elif nom_kpi == "Efficacité du Capital Économique (RoRC)":
                capital_requis = max(1000.0, p99_macro - (prime_mod - mu_perte))
                rorc = (prime_mod - mu_perte) / capital_requis
                return float(max(0.0, rorc * 100))
                
            elif nom_kpi == "Indice de Prudence (Marge de Sécurité)":
                return float((prime_mod - mu_perte) / mu_perte * 100)
                
            return 0.0

        scores = {
            "Parametriks Engine (IA)": (prime_moy_parametriks, ruine_parametriks, calculate_model_score(prime_moy_parametriks, ruine_parametriks, kpi_choisi)),
            "XGBoost Monotone (ML)": (prime_moy_xgboost, ruine_xgboost, calculate_model_score(prime_moy_xgboost, ruine_xgboost, kpi_choisi)),
            "GLM Tweedie (Standard)": (prime_moy_tweedie, ruine_tweedie, calculate_model_score(prime_moy_tweedie, ruine_tweedie, kpi_choisi)),
            "Découplé Fréquence/Sévérité": (prime_moy_freq_sev, ruine_freq_sev, calculate_model_score(prime_moy_freq_sev, ruine_freq_sev, kpi_choisi))
        }

        best_model = max(scores, key=lambda k: scores[k][2])
        unit = "pts" if "Sharpe" in kpi_choisi else "%"
        
        st.success(f"**Modèle optimal via {kpi_choisi} : {best_model}**")
        st.write(f"*Score optimal obtenu :* **{scores[best_model][2]:,.1f} {unit}**")

        # Tableau des Modèles 
        df_macro_models = pd.DataFrame({
            "Architecture Modèle": list(scores.keys()),
            "Prime Moyenne Proposée (€)": [f"{v[0]:,.2f} €" for v in scores.values()],
            "Probabilité de Ruine Réduite": [f"{v[1]*100:.2f} %" for v in scores.values()],
            "Score Risque/Rendement": [f"{v[2]*100:.1f} / 100" for v in scores.values()],
            "Statut du Modèle": ["Meilleur" if k == best_model else "Aligné" for k in scores.keys()]
        })
        
        st.dataframe(df_macro_models, hide_index=True, use_container_width=True)
        # graphique à barres des Primes vs P95/P99 pour illustrer la décision
        st.markdown("#### Confrontation des Tarifs Modèles aux Seuils de Choc stochastiques")
        p95_macro = np.percentile(trajectoires_pertes, 95)
        p99_macro = np.percentile(trajectoires_pertes, 99)

        fig_macro_pricing = go.Figure()
        # Barres de primes des modèles
        fig_macro_pricing.add_trace(go.Bar(
            x=list(scores.keys()), 
            y=[v[0] for v in scores.values()],
            name="Niveau de Prime Émise",
            marker_color=['#2ecc71' if k == best_model else '#34495e' for k in scores.keys()]
        ))
        # Lignes de référence des chocs
        fig_macro_pricing.add_trace(go.Scatter(x=list(scores.keys()), y=[p95_macro]*4, mode='lines', name='Seuil de Choc P95 (Décennal)', line=dict(color='#f39c12', dash='dash')))
        fig_macro_pricing.add_trace(go.Scatter(x=list(scores.keys()), y=[p99_macro]*4, mode='lines', name='Seuil de Choc P99 (Solvabilité II)', line=dict(color='#e74c3c', width=2)))
        
        fig_macro_pricing.update_layout(
            title="Comparatif des Tarifications vs Niveaux de Sinistres Catastrophes",
            yaxis_title="Montants (€)",
            height=350,
            margin=dict(l=20, r=20, t=50, b=20)
        )
        st.plotly_chart(fig_macro_pricing, use_container_width=True)
        st.caption("💡 **Aide visuelle à la décision :** Un modèle performant doit proposer une barre de prime qui s'approche ou dépasse la ligne orange (P95) sans sur-tarifer inutilement au-delà de la ligne rouge (P99), sous peine de perdre sa compétitivité commerciale.")

        st.markdown("---")
        st.subheader("3. Densité des Trajectoires & Volatilité des Pertes")
        
        # Histogramme des trajectoires
        df_trajectoires = pd.DataFrame({"Pertes": trajectoires_pertes})
        fig_hist = px.histogram(
            df_trajectoires, x="Pertes", nbins=50,
            title="Répartition des Pertes Globales Générées par Monte Carlo",
            color_discrete_sequence=['#2980b9']
        )
        fig_hist.add_vline(x=market_premium_input, line_width=3, line_dash="dash", line_color="black", annotation_text="Prime Marché Actuelle")
        fig_hist.add_vline(x=prime_moy_parametriks, line_width=2, line_color="#2ecc71", annotation_text="Cible Parametriks")
        fig_hist.update_layout(xaxis_title="Sinistralité Agrégée par Scénario (€)", yaxis_title="Occurrences", height=300)
        st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.header("Optimisation génétique du modèle")
        st.write(f"Optimisation de l'ADN de l'agent face au profil de risque **{preset_choice}** pour maximiser le critère : **{kpi_choisi}**.")
        
        st.info(f" **Configuration active :** {total_generations} générations | {population_size_input} individus par vague | Taux de Mutation : {mutation_rate_input*100:.0f}%")
        with st.spinner("Brassage génétique, sélection de l'élite et calcul des mutations stochastiques..."):
            optimizer = EvolutionaryPricingOptimizer(
                simulation_service=mc_engine,
                n_generations=total_generations,
                population_size=population_size_input,
                mutation_rate=mutation_rate_input,
                contract_text=contract_text_input,
                market_premium=market_premium_input,
                historical_lr=historical_lr_input,
                exposure_count=exposure_input,
            )
            results = optimizer.optimize(
                kpi_director=kpi_choisi,
            )
            
        learning_curve = results["learning_curve"]
        best_genes = results["best_genes"]
        st.success(f" L'agent a stabilisé sa tarification d'équilibre à : **{results['optimized_premium']:,.2f} €**")
        
        # Grille de métriques avancées du Chromosome Élu
        with st.container(border=True):
            st.subheader("Chromosome optimal")
            genes = results["best_genes"]
            c1, c2, c3 = st.columns(3)
            c1.metric("Marge de Sécurité Globale", f"{genes['safety_margin']:.3f}x")
            c2.metric("Seuil Quantile (Target VaR)", f"{genes['risk_quantile']*100:.2f} %")
            c3.metric("Stress Factor Macro", f"{genes['macro_stress_factor']:.2f}x")

# Double graphique d'analyse de la convergence
        g1, g2 = st.columns(2)
        
        with g1:
            fig_curve = px.line(
                x=range(1, len(learning_curve) + 1), y=learning_curve,
                labels={"x": "Générations", "y": "Score de Fitness"},
                title="Courbe d'Apprentissage",
                markers=True
            )
            fig_curve.update_traces(line_color='#2563EB', marker=dict(size=4))
            st.plotly_chart(fig_curve, use_container_width=True)
            
        with g2:
            # CORRECTION : Adaptation aux nouveaux gènes de simulation réels
            fig_genes = go.Figure(go.Bar(
                x=["Marge Sécurité", "Quantile Cible", "Stress Macro"],
                y=[best_genes['safety_margin'], best_genes['risk_quantile'], best_genes['macro_stress_factor']],
                marker_color=['#1E3A8A', '#10B981', '#F59E0B']
            ))
            fig_genes.update_layout(title="Configuration des Gènes Optimisés", height=325)
            st.plotly_chart(fig_genes, use_container_width=True)

        st.markdown("---")
        st.subheader("Trajectoire du modèle")
        st.write("Observez comment l'agent Parametriks se positionne face aux modèles classiques du marché à chaque étape clé de sa mutation génétique.")

        paliers = [i for i in range(0, len(learning_curve), 10)]
        if (len(learning_curve) - 1) not in paliers:
            paliers.append(len(learning_curve) - 1)
            
        titles_tabs = [f"Génération {p if p > 0 else 1}" for p in paliers]
        tabs = st.tabs(titles_tabs)

        # Calcul de la prime de base native générée par votre agent du backend
        from backend import ParametriksPricingAgent
        native_agent = ParametriksPricingAgent(mc_engine)
        base_parametriks_premium = native_agent.analyze_contract_and_price(
                contract_text=contract_text_input,
                market_premium=Decimal(str(market_premium_input)),
                historical_loss_ratio=historical_lr_input,
                exposure_count=exposure_input
            )["actionable_decision"]["proposed_premium"]

        mu_perte = market_premium_input * historical_lr_input
        sigma_perte = mu_perte * (0.3 if "limitation" in contract_text_input.lower() else 0.6)
        shape = np.sqrt(np.log(1 + (sigma_perte / mu_perte) ** 2))
        scale = mu_perte / np.sqrt(1 + (sigma_perte / mu_perte) ** 2)
        trajectoires_pertes = np.random.lognormal(mean=np.log(scale), sigma=shape, size=1000)
        
        p95_macro = np.percentile(trajectoires_pertes, 95)
        p99_macro = np.percentile(trajectoires_pertes, 99)
        tvar_95_macro = np.mean(trajectoires_pertes[trajectoires_pertes >= p95_macro])

        for idx, palier in enumerate(paliers):
            with tabs[idx]:
                st.markdown(f"##### Comparaison des modèles — **Génération {palier if palier > 0 else 1}**")
                
                # CORRECTION : Trajectoire d'évolution reconstruite à partir des nouveaux gènes
                ratio_evolution = (palier + 1) / len(learning_curve)
                
                # Progression linéaire pédagogique vers les hyperparamètres champions
                interim_margin = 1.0 + (best_genes['safety_margin'] - 1.0) * ratio_evolution
                interim_stress = 1.0 + (best_genes['macro_stress_factor'] - 1.0) * ratio_evolution
                
                # Reconstruction exacte de la formule de tarification de geneticalg.py
                has_limitation = "limitation" in contract_text_input.lower()
                prime_parametriks_interim = base_parametriks_premium * interim_margin * (interim_stress if not has_limitation else 1.0)
                
                prime_moy_xgboost = market_premium_input * (1.18 if preset_choice in ["Auto", "Tech"] else 1.45)
                prime_moy_tweedie = market_premium_input * (1.05 if preset_choice == "Auto" else 1.12)
                prime_moy_freq_sev = market_premium_input * 1.20

                ruine_param = np.mean(trajectoires_pertes > prime_parametriks_interim)
                ruine_xgb = np.mean(trajectoires_pertes > prime_moy_xgboost)
                ruine_tweedie = np.mean(trajectoires_pertes > prime_moy_tweedie)
                ruine_freq_sev = np.mean(trajectoires_pertes > prime_moy_freq_sev)

                def evaluer_modele_kpi_pur(prime_mod, proba_ruine):
                    marge_brute = (prime_mod - mu_perte) / prime_mod
                    
                    if kpi_choisi == "Ratio Risque/Rendement (Sharpe Adapté)":
                        return max(0.0, marge_brute - (proba_ruine * 2.5))
                        
                    elif kpi_choisi == "Résilience aux Chocs de Queue (TVaR 95%)":
                        return float(min(100.0, (prime_mod / tvar_95_macro) * 100))
                        
                    elif kpi_choisi == "Efficacité du Capital Économique (RoRC)":
                        capital_requis = max(1000.0, p99_macro - (prime_mod - mu_perte))
                        rorc = (prime_mod - mu_perte) / capital_requis
                        return float(max(0.0, rorc * 100))
                        
                    elif kpi_choisi == "Indice de Prudence (Marge de Sécurité)":
                        return float((prime_mod - mu_perte) / mu_perte * 100)
                        
                    return 0.0

                scores_kpi_purs = {
                    "Parametriks (En Mutation)": evaluer_modele_kpi_pur(prime_parametriks_interim, ruine_param),
                    "XGBoost Monotone (ML)": evaluer_modele_kpi_pur(prime_moy_xgboost, ruine_xgb),
                    "GLM Tweedie (Standard)": evaluer_modele_kpi_pur(prime_moy_tweedie, ruine_tweedie),
                    "Découplé Fréquence/Sévérité": evaluer_modele_kpi_pur(prime_moy_freq_sev, ruine_freq_sev)
                }

                champion_kpi_local = max(scores_kpi_purs, key=scores_kpi_purs.get)               
                unite_kpi = "pts" if "Sharpe" in kpi_choisi else "%"

                df_palier_models = pd.DataFrame({
                    "Architecture Modèle": list(scores_kpi_purs.keys()),
                    "Tarif Proposé (€)": [f"{prime_parametriks_interim:,.2f} €", f"{prime_moy_xgboost:,.2f} €", f"{prime_moy_tweedie:,.2f} €", f"{prime_moy_freq_sev:,.2f} €"],
                    "Probabilité de Ruine": [f"{ruine_param*100:.2f} %", f"{ruine_xgb*100:.2f} %", f"{ruine_tweedie*100:.2f} %", f"{ruine_freq_sev*100:.2f} %"],
                    f"Score {kpi_choisi}": [f"{v:,.2f} {unite_kpi}" for v in scores_kpi_purs.values()],
                    "Verdict Arbitrage": ["🏆 Optimal" if k == champion_kpi_local else "❌ Rejeté" for k in scores_kpi_purs.keys()]
                })
                
                st.dataframe(df_palier_models, hide_index=True, use_container_width=True)
                
                fig_palier_pricing = go.Figure()
                fig_palier_pricing.add_trace(go.Bar(
                    x=["Parametriks", "XGBoost", "GLM Tweedie", "Freq/Sev"], 
                    y=[prime_parametriks_interim, prime_moy_xgboost, prime_moy_tweedie, prime_moy_freq_sev],
                    name="Niveau de Prime Émise",
                    marker_color=['#2563EB' if k == "Parametriks (En Mutation)" else '#475569' for k in scores_kpi_purs.keys()]
                ))
                fig_palier_pricing.add_trace(go.Scatter(x=["Parametriks", "XGBoost", "GLM Tweedie", "Freq/Sev"], y=[p95_macro]*4, mode='lines', name='Choc P95 (Décennal)', line=dict(color='#F59E0B', dash='dash')))
                fig_palier_pricing.add_trace(go.Scatter(x=["Parametriks", "XGBoost", "GLM Tweedie", "Freq/Sev"], y=[p99_macro]*4, mode='lines', name='Choc P99 (Solvabilité II)', line=dict(color='#EF4444', width=2)))
                
                fig_palier_pricing.update_layout(
                    title=f"Positionnement des Tarifs vs Seuils de Ruine à la Génération {palier if palier > 0 else 1}",
                    yaxis_title="Montants (€)", height=280, margin=dict(l=20, r=20, t=40, b=20)
                )
                st.plotly_chart(fig_palier_pricing, use_container_width=True)
else:
    # page vide
    st.info("💡 Sélectionnez vos paramètres dans le panneau de gauche et cliquez sur **Lancer le Benchmark Global** pour déployer l'analyse unifiée.")