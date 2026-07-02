# Parametriks AI Pricing Cockpit
---

## Project Overview

The **Parametriks AI Pricing Cockpit** allows insurers, underwriters, and actuarial data scientists to evaluate, bench, and simulate complex risk portfolios. By ingesting policy text and technical benchmarks, the system runs high-fidelity stochastic simulations to automate underwriting decisions while confronting traditional financial models against AI-driven frameworks.

### Key Pillars
*   **Explainable AI (XAI) Underwriting:** Decodes policy wording, identifies technical clauses (e.g., safety caps), and provides real-time automated decisions (`SUBSCRIBE`, `REFER`, `DECLINE`).
*   **Multi-Model Confrontation:** Compares traditional generalized linear models (**GLM Tweedie**), decoupled actuarial methods (**Frequency/Severity**), and advanced machine learning (**Monotone XGBoost**).
*   **Stochastic Risk Assessment:** Executes large-scale **Monte Carlo simulations** to map tail risks ($VaR_{99}$, $TVaR$), calculate ruin probabilities, and declare the optimal pricing model.

---

## Core Features & Capabilities

### 1. Dual Simulation Framework
The cockpit splits operational analysis into two scalable workflows:
*   **Unitary Contract Analysis:** In-depth, individual policy underwriting. Extracts semantic risk features, computes stochastically-adjusted loss distributions, and performs reverse-engineered ML reproductions (Quantile Clone Regression).
*   **Multi-Simulation Stress-Testing:** Macro-level portfolio stress-testing. Runs up to 5 000 aggregate parallel trajectories to track capital volatility, project combined ratios, and quantify global technical ruin probabilities.

### 2. Built-in Portfolio Actuarial Presets
The system includes native, empirical risk profile presets derived from real-world market distributions:
*   **Auto:** High frequency, low severity ($\Gamma$ Gamma distribution).
*   **Industrie:** Medium frequency, high dispersion (Lognormal distribution).
*   **Cyber:** Low frequency, extreme systemic tail risk ($\mathcal{P}$ Pareto distribution).
*   **Tech:** Balanced E&O profile with controlled volatility (Lognormal distribution).
*   **RC_Generale:** Low frequency, long-tail claim development ($\mathcal{P}$ Pareto distribution).

### 3. Automated Arbitrage (The Optimal Selector)
For macro stress-testing, the cockpit applies an automated decision metric resembling an actuarial *Sharpe Ratio*. It dynamically penalizes models vulnerable to catastrophic tail risk and rewards efficient technical margins, naming the **Champion Model** best suited for the specific risk distribution.

---

## Getting Started & Local Installation

### Prerequisites
Ensure you have **Python 3.10+** installed on your system. 

### 1. Clone the Repository
```bash
git clone [https://github.com/your-username/parametriks-pricing-cockpit.git](https://github.com/your-username/parametriks-pricing-cockpit.git)
cd petriks-pricing-cockpit
```

### 2. Install the Requirements
Type in in your terminal : 
```bash
pip install -r requirements.txt
```

### 3. Run the app using streamlit
```bash
streamlit run app.py
```
The app will open automatically in your default browser
