import numpy as np
import random
from decimal import Decimal
# Import direct des classes de votre dépôt
from backend import ParametriksPricingAgent, ScenarioSimulationService

class EvolutionaryPricingOptimizer:
    """
    Evolutionary Pricing Optimization Engine (Genetic Algorithm).
    Optimizes the ParametriksPricingAgent behavior by adjusting simulation parameters
    over N generations under simulated market elasticity and capital constraints.
    """
    def __init__(self, simulation_service: ScenarioSimulationService, n_generations=30, population_size=20, mutation_rate=0.15, contract_text="", market_premium=0, historical_lr=0, exposure_count=0):
        self.simulation_service = simulation_service
        self.n_generations = n_generations
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.contract_text_input = contract_text
        self.market_premium_input = market_premium
        self.historical_lr_input = historical_lr
        self.exposure_input = exposure_count

    def _generate_random_simulation_genes(self):
        """ 
        Generates a chromosome representing adjustable simulation/pricing dimensions.
        Adjust these keys based on what your ScenarioSimulationService exposes 
        (e.g., loss loading, security quantiles, or volatility coefficients).
        """
        return {
            "safety_margin": random.uniform(1.10, 2.50),       # Multiplier for stochastic losses
            "risk_quantile": random.uniform(0.90, 0.995),      # Target VaR/TVaR safety threshold
            "macro_stress_factor": random.uniform(1.0, 1.40)   # Macro degradation multiplier
        }

    def _simulate_market_elasticity(self, agent_premium, market_premium):
        """ Models market elasticity to benchmark commercial conversion against the market """
        premium_ratio = float(agent_premium) / float(market_premium)
        if premium_ratio <= 1.0:
            return 0.95
        else:
            return float(1.0 / (1.0 + np.exp(5 * (premium_ratio - 1.30))))

    def _evaluate_fitness(self, chromosome, kpi_name):
        """ Fitness Function: Plugs genes into the native Parametriks workflow """
        
        parametriks_agent = ParametriksPricingAgent(self.simulation_service)

        agent_unitary_premium = parametriks_agent.analyze_contract_and_price(contract_text=self.contract_text_input,
                market_premium=Decimal(str(self.market_premium_input)),
                historical_loss_ratio=self.historical_lr_input,
                exposure_count=self.exposure_input
            )["actionable_decision"]["proposed_premium"]
        

        adjusted_unitary_premium = agent_unitary_premium * chromosome["safety_margin"] * (chromosome["macro_stress_factor"] if "limitation" not in self.contract_text_input.lower() else 1.0)
        agent_portfolio_premium = adjusted_unitary_premium * self.exposure_input

        mean_loss = float(self.market_premium_input) * self.historical_lr_input
        loss_standard_deviation = mean_loss * (0.3 if "limitation" in self.contract_text_input.lower() else 0.6)
        
        lognormal_shape = np.sqrt(np.log(1 + (loss_standard_deviation / mean_loss) ** 2))
        lognormal_scale = mean_loss / np.sqrt(1 + (loss_standard_deviation / mean_loss) ** 2)
        
        loss_trajectories = np.random.lognormal(mean=np.log(lognormal_scale), sigma=lognormal_shape, size=1000)
        stochastic_mean_loss = np.mean(loss_trajectories)
        
        target_quantile = min(0.999, max(0.80, chromosome["risk_quantile"]))
        var_target_macro = np.percentile(loss_trajectories, target_quantile * 100)
        var_99_macro = np.percentile(loss_trajectories, 99)
        tvar_95_macro = np.mean(loss_trajectories[loss_trajectories >= np.percentile(loss_trajectories, 95)])
        ruin_probability = np.mean(loss_trajectories > adjusted_unitary_premium)
        gross_margin = (agent_portfolio_premium - (stochastic_mean_loss * self.exposure_input)) / agent_portfolio_premium
        
        if kpi_name == "Ratio Risque/Rendement (Sharpe Adapté)":
            technical_score = max(0.0, gross_margin - (ruin_probability * 2.5))
        elif kpi_name == "Résilience aux Chocs de Queue (TVaR 95%)":
            technical_score = float(min(100.0, (adjusted_unitary_premium / tvar_95_macro) * 100))
        elif kpi_name == "Efficacité du Capital Économique (RoRC)":
            economic_capital_required = max(1000.0, var_99_macro - (adjusted_unitary_premium - mean_loss))
            technical_score = float(max(0.0, ((adjusted_unitary_premium - mean_loss) / economic_capital_required) * 100))
        else:
            technical_score = float((adjusted_unitary_premium - mean_loss) / mean_loss * 100)

        conversion_rate = self._simulate_market_elasticity(adjusted_unitary_premium, self.market_premium_input)

        return max(0.0, technical_score * conversion_rate)

    def _crossover(self, parent1, parent2):
        alpha = random.random()
        child = {}
        for gene in parent1.keys():
            child[gene] = alpha * parent1[gene] + (1 - alpha) * parent2[gene]
        return child

    def _mutate(self, chromosome):
        for gene in chromosome.keys():
            if random.random() < self.mutation_rate:
                chromosome[gene] *= random.uniform(0.90, 1.10)
                if gene == "risk_quantile":
                    chromosome[gene] = min(0.995, max(0.80, chromosome[gene]))
        return chromosome

    def optimize(self, kpi_director):
        population = [self._generate_random_simulation_genes() for _ in range(self.population_size)]
        history_best_scores = []

        for generation in range(self.n_generations):
            evaluated_population = []
            for genes in population:
                score = self._evaluate_fitness(
                    genes, kpi_director
                )
                evaluated_population.append((genes, score))
            
            evaluated_population.sort(key=lambda x: x[1], reverse=True)
            history_best_scores.append(evaluated_population[0][1])
            
            elites = [individual[0] for individual in evaluated_population[:self.population_size // 4]]
            
            next_generation = list(elites)
            while len(next_generation) < self.population_size:
                parent1, parent2 = random.sample(elites, 2)
                child = self._crossover(parent1, parent2)
                child = self._mutate(child)
                next_generation.append(child)
                
            population = next_generation

        best_genes = evaluated_population[0][0]
        
        # Calcul final de la prime recommandée
        final_agent = ParametriksPricingAgent(self.simulation_service)
        base_premium = final_agent.analyze_contract_and_price(contract_text=self.contract_text_input,
                market_premium=Decimal(str(self.market_premium_input)),
                historical_loss_ratio=self.historical_lr_input,
                exposure_count=self.exposure_input
            )["actionable_decision"]["proposed_premium"]
        optimized_premium = base_premium * best_genes["safety_margin"] * (best_genes["macro_stress_factor"] if "limitation" not in self.contract_text_input.lower() else 1.0)

        return {
            "best_genes": best_genes,
            "optimized_premium": optimized_premium,
            "learning_curve": history_best_scores
        }