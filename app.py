import streamlit as st
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from random import randint, random
from copy import deepcopy
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Algorithm abbreviations mapping
ALGO_ABBREVIATIONS = {
    "Genetic Algorithm": "GA",
    "Particle Swarm": "PSO",
    "Ant Colony": "ACO",
    "Grey Wolf": "GWO",
    "Whale Optimization": "WOA",
    "Hybrid GWO-WOA": "HYB"
}

# Optimization Algorithms with improved implementations
class OptimizationAlgorithms:
    
    @staticmethod
    def genetic_algorithm(job_sizes, params):
        """Improved Genetic Algorithm implementation"""
        start_time = time.time()
        
        population_size = params['population_size']
        max_generations = params['max_generations']
        crossover_rate = params.get('crossover_rate', 0.8)
        mutation_rate = params.get('mutation_rate', 0.1)
        
        # Initialize population with permutations
        population = [np.random.permutation(job_sizes) for _ in range(population_size)]
        best_fitness = float('inf')
        
        for generation in range(max_generations):
            # Evaluate fitness (makespan calculation)
            fitness = [OptimizationAlgorithms.calculate_makespan(ind) for ind in population]
            current_best = min(fitness)
            
            if current_best < best_fitness:
                best_fitness = current_best
                best_solution = population[fitness.index(current_best)]
            
            # Tournament selection
            selected = []
            for _ in range(population_size):
                candidates = [randint(0, population_size-1) for _ in range(2)]
                winner = min(candidates, key=lambda x: fitness[x])
                selected.append(population[winner])
            
            # Ordered crossover (OX)
            new_population = []
            for i in range(0, population_size, 2):
                parent1, parent2 = selected[i], selected[i+1]
                
                if random() < crossover_rate:
                    # Create offspring using OX
                    size = len(parent1)
                    a, b = sorted([randint(0, size-1), randint(0, size-1)])
                    
                    # Initialize offspring with -1
                    child1 = np.full(size, -1)
                    child2 = np.full(size, -1)
                    
                    # Copy the segment between a and b
                    child1[a:b+1] = parent1[a:b+1]
                    child2[a:b+1] = parent2[a:b+1]
                    
                    # Fill remaining positions
                    ptr1 = ptr2 = (b + 1) % size
                    for j in range(size):
                        val1 = parent2[(b + 1 + j) % size]
                        val2 = parent1[(b + 1 + j) % size]
                        
                        if val1 not in child1:
                            child1[ptr1] = val1
                            ptr1 = (ptr1 + 1) % size
                            
                        if val2 not in child2:
                            child2[ptr2] = val2
                            ptr2 = (ptr2 + 1) % size
                    
                    new_population.extend([child1, child2])
                else:
                    new_population.extend([parent1, parent2])
            
            # Swap mutation
            for i in range(population_size):
                if random() < mutation_rate:
                    a, b = randint(0, len(job_sizes)-1), randint(0, len(job_sizes)-1)
                    new_population[i][a], new_population[i][b] = new_population[i][b], new_population[i][a]
            
            population = new_population
        
        runtime = time.time() - start_time
        return best_fitness, runtime

    @staticmethod
    def particle_swarm(job_sizes, params):
        """Improved PSO implementation"""
        start_time = time.time()
        
        swarm_size = params['swarm_size']
        max_iterations = params['max_iterations']
        w = params.get('inertia_weight', 0.7)
        c1 = params.get('cognitive_weight', 1.5)
        c2 = params.get('social_weight', 1.5)
        
        # Initialize particles as permutations
        particles = [np.random.permutation(job_sizes) for _ in range(swarm_size)]
        velocities = [np.zeros(len(job_sizes)) for _ in range(swarm_size)]
        
        personal_best = deepcopy(particles)
        personal_best_fitness = [OptimizationAlgorithms.calculate_makespan(p) for p in particles]
        
        global_best = min(particles, key=lambda x: OptimizationAlgorithms.calculate_makespan(x))
        global_best_fitness = OptimizationAlgorithms.calculate_makespan(global_best)
        
        for _ in range(max_iterations):
            for i in range(swarm_size):
                # Update velocity (using swap operator probabilities)
                for d in range(len(job_sizes)):
                    r1, r2 = random(), random()
                    
                    # Cognitive component
                    if particles[i][d] != personal_best[i][d]:
                        cognitive = c1 * r1 * (1 if random() < 0.5 else -1)
                    else:
                        cognitive = 0
                    
                    # Social component
                    if particles[i][d] != global_best[d]:
                        social = c2 * r2 * (1 if random() < 0.5 else -1)
                    else:
                        social = 0
                    
                    velocities[i][d] = w * velocities[i][d] + cognitive + social
                
                # Update position (apply swaps based on velocity)
                for d in range(len(job_sizes)):
                    if random() < abs(velocities[i][d]):
                        swap_with = randint(0, len(job_sizes)-1)
                        particles[i][d], particles[i][swap_with] = particles[i][swap_with], particles[i][d]
                
                # Evaluate new position
                current_fitness = OptimizationAlgorithms.calculate_makespan(particles[i])
                
                # Update personal best
                if current_fitness < personal_best_fitness[i]:
                    personal_best[i] = deepcopy(particles[i])
                    personal_best_fitness[i] = current_fitness
                    
                    # Update global best
                    if current_fitness < global_best_fitness:
                        global_best = deepcopy(particles[i])
                        global_best_fitness = current_fitness
        
        runtime = time.time() - start_time
        return global_best_fitness, runtime
    
    @staticmethod
    def ant_colony(job_sizes, params):
        """Improved ACO implementation"""
        start_time = time.time()
        
        n_ants = params['n_ants']
        n_iterations = params['n_iterations']
        alpha = params.get('alpha', 1.0)  # Pheromone importance
        beta = params.get('beta', 2.0)   # Heuristic importance
        rho = params.get('rho', 0.1)    # Evaporation rate
        q = params.get('q', 1.0)        # Pheromone deposit factor
        
        n_jobs = len(job_sizes)
        heuristic = 1 / np.array(job_sizes)  # Smaller jobs are more desirable
        
        # Initialize pheromone matrix
        pheromone = np.ones((n_jobs, n_jobs)) * 0.1
        
        best_solution = None
        best_fitness = float('inf')
        
        for iteration in range(n_iterations):
            solutions = []
            fitness_values = []
            
            for ant in range(n_ants):
                # Construct solution using pheromone and heuristic
                visited = [False] * n_jobs
                solution = []
                
                # Start with a random job
                current = randint(0, n_jobs-1)
                solution.append(current)
                visited[current] = True
                
                while len(solution) < n_jobs:
                    # Calculate probabilities for next job
                    probabilities = []
                    total = 0.0
                    
                    for j in range(n_jobs):
                        if not visited[j]:
                            prob = (pheromone[current][j] ** alpha) * (heuristic[j] ** beta)
                            probabilities.append((j, prob))
                            total += prob
                    
                    # Select next job
                    if total > 0:
                        probabilities = [(j, p/total) for j, p in probabilities]
                        probabilities.sort(key=lambda x: x[1], reverse=True)
                        
                        # Roulette wheel selection
                        r = random()
                        cum_prob = 0.0
                        for j, prob in probabilities:
                            cum_prob += prob
                            if r <= cum_prob:
                                current = j
                                break
                    else:
                        # If all probabilities are zero, select randomly
                        unvisited = [j for j in range(n_jobs) if not visited[j]]
                        current = unvisited[randint(0, len(unvisited)-1)]
                    
                    solution.append(current)
                    visited[current] = True
                
                # Evaluate solution
                solution_fitness = OptimizationAlgorithms.calculate_makespan([job_sizes[i] for i in solution])
                solutions.append(solution)
                fitness_values.append(solution_fitness)
                
                # Update best solution
                if solution_fitness < best_fitness:
                    best_fitness = solution_fitness
                    best_solution = solution
            
            # Update pheromones
            pheromone *= (1 - rho)  # Evaporation
            
            # Add new pheromones
            for ant in range(n_ants):
                solution = solutions[ant]
                fitness = fitness_values[ant]
                
                if fitness > 0:
                    delta = q / fitness
                    
                    for i in range(len(solution)-1):
                        pheromone[solution[i]][solution[i+1]] += delta
                        pheromone[solution[i+1]][solution[i]] += delta  # Symmetric
        
        runtime = time.time() - start_time
        return best_fitness, runtime
    
    @staticmethod
    def grey_wolf(job_sizes, params):
        """Improved Grey Wolf Optimizer implementation"""
        start_time = time.time()
        
        population_size = params['population_size']
        max_iterations = params['max_iterations']
        
        # Initialize wolves as permutations
        wolves = [np.random.permutation(job_sizes) for _ in range(population_size)]
        fitness = [OptimizationAlgorithms.calculate_makespan(wolf) for wolf in wolves]
        
        # Sort wolves: alpha, beta, delta, omega
        sorted_indices = np.argsort(fitness)
        alpha = deepcopy(wolves[sorted_indices[0]])
        beta = deepcopy(wolves[sorted_indices[1]])
        delta = deepcopy(wolves[sorted_indices[2]])
        
        alpha_fitness = fitness[sorted_indices[0]]
        
        for iteration in range(max_iterations):
            a = 2 - iteration * (2 / max_iterations)  # Decreases linearly from 2 to 0
            
            for i in range(population_size):
                # Update position using alpha, beta, delta
                r1, r2 = random(), random()
                A1 = 2 * a * r1 - a
                C1 = 2 * r2
                
                D_alpha = abs(C1 * alpha - wolves[i])
                X1 = alpha - A1 * D_alpha
                
                r1, r2 = random(), random()
                A2 = 2 * a * r1 - a
                C2 = 2 * r2
                
                D_beta = abs(C2 * beta - wolves[i])
                X2 = beta - A2 * D_beta
                
                r1, r2 = random(), random()
                A3 = 2 * a * r1 - a
                C3 = 2 * r2
                
                D_delta = abs(C3 * delta - wolves[i])
                X3 = delta - A3 * D_delta
                
                # New position is average of X1, X2, X3
                new_position = (X1 + X2 + X3) / 3
                
                # Convert to permutation (nearest permutation)
                wolves[i] = OptimizationAlgorithms.nearest_permutation(new_position)
                
                # Evaluate new position
                current_fitness = OptimizationAlgorithms.calculate_makespan(wolves[i])
                
                # Update alpha, beta, delta
                if current_fitness < alpha_fitness:
                    delta = deepcopy(beta)
                    beta = deepcopy(alpha)
                    alpha = deepcopy(wolves[i])
                    alpha_fitness = current_fitness
                elif current_fitness < OptimizationAlgorithms.calculate_makespan(beta):
                    delta = deepcopy(beta)
                    beta = deepcopy(wolves[i])
                elif current_fitness < OptimizationAlgorithms.calculate_makespan(delta):
                    delta = deepcopy(wolves[i])
        
        runtime = time.time() - start_time
        return alpha_fitness, runtime
    
    @staticmethod
    def whale_optimization(job_sizes, params):
        """Improved Whale Optimization Algorithm implementation"""
        start_time = time.time()
        
        population_size = params['population_size']
        max_iterations = params['max_iterations']
        
        # Initialize whales as permutations
        whales = [np.random.permutation(job_sizes) for _ in range(population_size)]
        fitness = [OptimizationAlgorithms.calculate_makespan(whale) for whale in whales]
        
        best_idx = np.argmin(fitness)
        best_whale = deepcopy(whales[best_idx])
        best_fitness = fitness[best_idx]
        
        for iteration in range(max_iterations):
            a = 2 - iteration * (2 / max_iterations)  # Decreases linearly from 2 to 0
            a2 = -1 + iteration * (-1 / max_iterations)  # For spiral updating
            
            for i in range(population_size):
                r = random()
                A = 2 * a * r - a
                C = 2 * r
                
                p = random()
                
                if p < 0.5:
                    if abs(A) < 1:
                        # Encircling prey
                        D = abs(C * best_whale - whales[i])
                        new_position = best_whale - A * D
                    else:
                        # Search for prey
                        rand_idx = randint(0, population_size-1)
                        rand_whale = whales[rand_idx]
                        D = abs(C * rand_whale - whales[i])
                        new_position = rand_whale - A * D
                else:
                    # Bubble-net attacking
                    b = 1  # Defines shape of spiral
                    l = (a2 - 1) * random() + 1
                    
                    D = abs(best_whale - whales[i])
                    new_position = D * np.exp(b * l) * np.cos(2 * np.pi * l) + best_whale
                
                # Convert to nearest permutation
                whales[i] = OptimizationAlgorithms.nearest_permutation(new_position)
                
                # Evaluate new position
                current_fitness = OptimizationAlgorithms.calculate_makespan(whales[i])
                
                # Update best solution
                if current_fitness < best_fitness:
                    best_whale = deepcopy(whales[i])
                    best_fitness = current_fitness
        
        runtime = time.time() - start_time
        return best_fitness, runtime
    
    @staticmethod
    def hybrid_grey_wolf_whale(job_sizes, params):
        """Hybrid Grey Wolf-Whale Optimization Algorithm implementation"""
        start_time = time.time()
        
        population_size = params['population_size']
        max_iterations = params['max_iterations']
        
        # Initialize population as permutations
        population = [np.random.permutation(job_sizes) for _ in range(population_size)]
        fitness = [OptimizationAlgorithms.calculate_makespan(ind) for ind in population]
        
        # Initialize alpha, beta, delta wolves (best three solutions)
        sorted_indices = np.argsort(fitness)
        alpha = deepcopy(population[sorted_indices[0]])
        beta = deepcopy(population[sorted_indices[1]])
        delta = deepcopy(population[sorted_indices[2]])
        
        alpha_fitness = fitness[sorted_indices[0]]
        
        for iteration in range(max_iterations):
            a = 2 - iteration * (2 / max_iterations)  # Decreases linearly from 2 to 0
            a2 = -1 + iteration * (-1 / max_iterations)  # For spiral updating
            
            for i in range(population_size):
                r = random()
                A = 2 * a * r - a
                C = 2 * r
                p = random()
                
                if p < 0.5:
                    if abs(A) < 1:
                        # Encircling prey (WOA)
                        D = abs(C * alpha - population[i])
                        new_position = alpha - A * D
                    else:
                        # Search for prey (WOA)
                        rand_idx = randint(0, population_size-1)
                        rand_whale = population[rand_idx]
                        D = abs(C * rand_whale - population[i])
                        new_position = rand_whale - A * D
                else:
                    # Bubble-net attacking (WOA)
                    b = 1  # Defines shape of spiral
                    l = (a2 - 1) * random() + 1
                    D = abs(alpha - population[i])
                    new_position = D * np.exp(b * l) * np.cos(2 * np.pi * l) + alpha
                
                # Grey Wolf component - update with alpha, beta, delta influence
                r1, r2 = random(), random()
                A1 = 2 * a * r1 - a
                C1 = 2 * r2
                D_alpha = abs(C1 * alpha - new_position)
                X1 = alpha - A1 * D_alpha
                
                r1, r2 = random(), random()
                A2 = 2 * a * r1 - a
                C2 = 2 * r2
                D_beta = abs(C2 * beta - new_position)
                X2 = beta - A2 * D_beta
                
                r1, r2 = random(), random()
                A3 = 2 * a * r1 - a
                C3 = 2 * r2
                D_delta = abs(C3 * delta - new_position)
                X3 = delta - A3 * D_delta
                
                # Combined new position (average of WOA and GWO components)
                new_position = (new_position + X1 + X2 + X3) / 4
                
                # Convert to nearest permutation
                population[i] = OptimizationAlgorithms.nearest_permutation(new_position)
                
                # Evaluate new position
                current_fitness = OptimizationAlgorithms.calculate_makespan(population[i])
                
                # Update alpha, beta, delta
                if current_fitness < alpha_fitness:
                    delta = deepcopy(beta)
                    beta = deepcopy(alpha)
                    alpha = deepcopy(population[i])
                    alpha_fitness = current_fitness
                elif current_fitness < OptimizationAlgorithms.calculate_makespan(beta):
                    delta = deepcopy(beta)
                    beta = deepcopy(population[i])
                elif current_fitness < OptimizationAlgorithms.calculate_makespan(delta):
                    delta = deepcopy(population[i])
        
        runtime = time.time() - start_time
        return alpha_fitness, runtime

    @staticmethod
    def calculate_makespan(schedule, num_machines=4):
        """Calculate makespan for a given schedule on multiple machines"""
        machine_times = [0] * num_machines
        for job in schedule:
            # Assign job to the machine with the least current load
            min_machine = np.argmin(machine_times)
            machine_times[min_machine] += job
        return max(machine_times)
    
    @staticmethod
    def nearest_permutation(vector):
        """Convert a continuous vector to a permutation"""
        # Get the order of the values in the vector
        ranked = np.argsort(np.argsort(vector))
        # Convert to 1-based permutation
        return ranked + 1

def load_job_sizes(file_path):
    """Load job sizes from a GoCJ dataset file"""
    try:
        with open(file_path, 'r') as f:
            job_sizes = [float(line.strip()) for line in f if line.strip()]
        return job_sizes
    except Exception as e:
        st.error(f"Error loading file {file_path}: {str(e)}")
        return []

def get_dataset_files():
    """Get all dataset files from the datasets folder"""
    if not os.path.exists("datasets"):
        os.makedirs("datasets")
        return []
    
    files = [f for f in os.listdir("datasets") if f.startswith("GoCJ_Dataset_") and f.endswith(".txt")]
    files.sort(key=lambda x: int(x.split('_')[2].split('.')[0]))  # Sort by job count
    return [os.path.join("datasets", f) for f in files]

def filter_datasets_by_size(dataset_files, size_category):
    """Filter dataset files based on the size category (small, medium, large)"""
    filtered_files = []
    
    for file_path in dataset_files:
        job_sizes = load_job_sizes(file_path)
        if not job_sizes:
            continue
        
        n_jobs = len(job_sizes)
        
        if size_category == "Small" and n_jobs < 300:
            filtered_files.append(file_path)
        elif size_category == "Medium" and 300 <= n_jobs < 700:
            filtered_files.append(file_path)
        elif size_category == "Large" and 700 <= n_jobs <= 1000:
            filtered_files.append(file_path)
    
    return filtered_files

def perform_statistical_analysis(results_df):
    st.subheader("🧪 Statistical Validation")
    
    # Ensure numeric data types for the metrics
    results_df['Makespan'] = pd.to_numeric(results_df['Makespan'], errors='coerce')
    results_df['Runtime (s)'] = pd.to_numeric(results_df['Runtime (s)'], errors='coerce')
    results_df['Total Cost (₹)'] = pd.to_numeric(results_df['Total Cost (₹)'], errors='coerce')
    
    # Drop any rows with missing values
    results_df = results_df.dropna(subset=['Makespan', 'Runtime (s)', 'Total Cost (₹)'])
    
    # Prepare data
    algorithms = results_df['Algorithm'].unique()
    metrics = ['Makespan', 'Runtime (s)', 'Total Cost (₹)']
    
    # Create tabs for each metric
    tab1, tab2, tab3 = st.tabs(metrics)
    
    with tab1:
        st.write("### Makespan Comparison")
        # ANOVA test for makespan
        groups = [results_df[results_df['Algorithm']==algo]['Makespan'].values for algo in algorithms]
        f_val, p_val = f_oneway(*groups)
        
        st.write(f"**ANOVA Test:** F-value = {f_val:.3f}, p-value = {p_val:.4f}")
        if p_val < 0.05:
            st.success("Significant differences exist between algorithms (p < 0.05)")
            
            # Post-hoc pairwise comparisons
            st.write("**Pairwise Comparisons (Tukey HSD):**")
            try:
                tukey = pairwise_tukeyhsd(
                    endog=results_df['Makespan'].values,
                    groups=results_df['Algorithm'].values,
                    alpha=0.05
                )
                st.text(str(tukey))
                
                # Plot the results
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=tukey.groupsunique,
                    y=tukey.meandiffs,
                    error_y=dict(
                        type='data',
                        array=tukey.confint[:, 1] - tukey.meandiffs,
                        arrayminus=tukey.meandiffs - tukey.confint[:, 0],
                        visible=True
                    ),
                    mode='markers'
                ))
                fig.update_layout(
                    title="Tukey HSD Test Results",
                    xaxis_title="Algorithm",
                    yaxis_title="Mean Difference",
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Could not perform Tukey HSD test: {str(e)}")
        else:
            st.warning("No significant differences found between algorithms")
    
    with tab2:
        st.write("### Runtime Comparison")
        # ANOVA test for runtime
        groups = [results_df[results_df['Algorithm']==algo]['Runtime (s)'].values for algo in algorithms]
        f_val, p_val = f_oneway(*groups)
        
        st.write(f"**ANOVA Test:** F-value = {f_val:.3f}, p-value = {p_val:.4f}")
        if p_val < 0.05:
            st.success("Significant differences exist between algorithms (p < 0.05)")
            
            # Post-hoc pairwise comparisons
            st.write("**Pairwise Comparisons (Tukey HSD):**")
            try:
                tukey = pairwise_tukeyhsd(
                    endog=results_df['Runtime (s)'].values,
                    groups=results_df['Algorithm'].values,
                    alpha=0.05
                )
                st.text(str(tukey))
            except Exception as e:
                st.error(f"Could not perform Tukey HSD test: {str(e)}")
        else:
            st.warning("No significant differences found between algorithms")
    
    with tab3:
        st.write("### Cost Comparison") 
        # ANOVA test for cost
        groups = [results_df[results_df['Algorithm']==algo]['Total Cost (₹)'].values for algo in algorithms]
        f_val, p_val = f_oneway(*groups)
        
        st.write(f"**ANOVA Test:** F-value = {f_val:.3f}, p-value = {p_val:.4f}")
        if p_val < 0.05:
            st.success("Significant differences exist between algorithms (p < 0.05)")
            
            # Post-hoc pairwise comparisons
            st.write("**Pairwise Comparisons (Tukey HSD):**")
            try:
                tukey = pairwise_tukeyhsd(
                    endog=results_df['Total Cost (₹)'].values,
                    groups=results_df['Algorithm'].values,
                    alpha=0.05
                )
                st.text(str(tukey))
            except Exception as e:
                st.error(f"Could not perform Tukey HSD test: {str(e)}")
        else:
            st.warning("No significant differences found between algorithms")

    # Add algorithm ranking
    st.write("### Algorithm Ranking")
    rank_df = pd.DataFrame({
        'Algorithm': algorithms,
        'Average Rank': [
            results_df[results_df['Algorithm']==algo]['Makespan'].rank().mean()
            for algo in algorithms
        ]
    }).sort_values('Average Rank')
    
    st.dataframe(rank_df.style.format({'Average Rank': '{:.2f}'}))

def plot_algorithm_behavior(results_df):
    st.subheader("🧠 Algorithm Behavior Analysis")
    
    # Convergence plots
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Exploration-Exploitation Balance", "Solution Quality Progression"))
    
    # Performance radar chart
    metrics = ['Makespan', 'Runtime', 'Cost']
    fig2 = go.Figure()
    
    for algo in ['Grey Wolf', 'Whale', 'Hybrid']:
        values = [
            results_df[results_df['Algorithm']==algo]['Makespan'].mean(),
            results_df[results_df['Algorithm']==algo]['Runtime (s)'].mean(),
            results_df[results_df['Algorithm']==algo]['Total Cost (₹)'].mean()
        ]
        fig2.add_trace(go.Scatterpolar(
            r=values,
            theta=metrics,
            fill='toself',
            name=ALGO_ABBREVIATIONS.get(algo, algo)
        ))
    
    fig2.update_layout(
        polar=dict(radialaxis=dict(visible=True)),
        showlegend=True,
        height=500
    )
    
    st.plotly_chart(fig2, use_container_width=True)
    
    # Add interpretation text
    st.write("""
    **Key Insights:**
    - The hybrid algorithm combines:
      - Grey Wolf's strong local search (exploitation)
      - Whale's spiral movement (exploration)
    - Adaptive switching balances these behaviors:
      - Early phase: More Whale-like exploration
      - Late phase: More Grey Wolf-like refinement
    """)

def main():
    st.set_page_config(page_title="Algorithm Efficiency Comparison", layout="wide")
    st.title("Optimization Algorithm Efficiency Comparison")
    
    # Get dataset files
    dataset_files = get_dataset_files()
    
    if not dataset_files:
        st.error("No dataset files found in the 'datasets' folder. Please add your GoCJ_Dataset_*.txt files there.")
        return
    
    # Initialize selected_files
    selected_files = []
    
    # Update the run_option radio button and dataset selection:
    run_option = st.radio("How would you like to run the algorithms?", 
                        ["Single File", "By Dataset Size", "All Dataset Sizes"])

    # Single File Option
    if run_option == "Single File":
        selected_file = st.selectbox("Select a dataset file", dataset_files)
        selected_files = [selected_file]
        
    # By Dataset Size Option
    elif run_option == "By Dataset Size":
        size_category = st.selectbox("Select Dataset Size", ["Small", "Medium", "Large"])
        selected_files = filter_datasets_by_size(dataset_files, size_category)
        if not selected_files:
            st.warning(f"No {size_category} datasets found.")
            return
            
    # All Dataset Sizes Option
    elif run_option == "All Dataset Sizes":
        selected_files = dataset_files
    
    # Algorithm parameters
    st.sidebar.header("Algorithm Parameters")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        population_size = st.slider("Population/Swarm Size", 10, 100, 30)
    with col2:
        max_iterations = st.slider("Max Iterations", 10, 500, 100)
    
    # Advanced parameters
    st.sidebar.header("Advanced Parameters")
    with st.sidebar.expander("Genetic Algorithm"):
        crossover_rate = st.slider("Crossover Rate", 0.1, 1.0, 0.8, key="ga_crossover")
        mutation_rate = st.slider("Mutation Rate", 0.01, 0.5, 0.1, key="ga_mutation")
    
    with st.sidebar.expander("Particle Swarm"):
        inertia_weight = st.slider("Inertia Weight", 0.1, 1.0, 0.7, key="pso_inertia")
        cognitive_weight = st.slider("Cognitive Weight", 0.1, 2.0, 1.5, key="pso_cognitive")
        social_weight = st.slider("Social Weight", 0.1, 2.0, 1.5, key="pso_social")
    
    with st.sidebar.expander("Ant Colony"):
        alpha = st.slider("Alpha (Pheromone)", 0.1, 5.0, 1.0, key="aco_alpha")
        beta = st.slider("Beta (Heuristic)", 0.1, 5.0, 2.0, key="aco_beta")
        rho = st.slider("Rho (Evaporation)", 0.01, 0.5, 0.1, key="aco_rho")
    
    # Cost parameters in Rupees
    st.sidebar.header("Cost Parameters (₹)")
    compute_cost_per_sec = st.sidebar.number_input("Compute Cost (₹/sec)", 0.01, 100.0, 0.05, step=0.01, format="%.2f")
    solution_cost_per_unit = st.sidebar.number_input("Solution Cost (₹/unit makespan)", 0.1, 100.0, 0.5, step=0.1, format="%.1f")
    
    # Prepare parameters for each algorithm (moved before the button check)
    params = {
        "Genetic Algorithm": {
            'population_size': population_size,
            'max_generations': max_iterations,
            'crossover_rate': crossover_rate,
            'mutation_rate': mutation_rate
        },
        "Particle Swarm": {
            'swarm_size': population_size,
            'max_iterations': max_iterations,
            'inertia_weight': inertia_weight,
            'cognitive_weight': cognitive_weight,
            'social_weight': social_weight
        },
        "Ant Colony": {
            'n_ants': population_size,
            'n_iterations': max_iterations,
            'alpha': alpha,
            'beta': beta,
            'rho': rho,
            'q': 1.0
        },
        "Grey Wolf": {
            'population_size': population_size,
            'max_iterations': max_iterations
        },
        "Whale Optimization": {
            'population_size': population_size,
            'max_iterations': max_iterations
        },
        "Hybrid GWO-WOA": {  # Parameters for the hybrid algorithm
            'population_size': population_size,
            'max_iterations': max_iterations
        }
    }

    if st.button("Run Algorithms", type="primary"):
        all_results = pd.DataFrame(columns=[
            "Dataset", "Jobs", "Algorithm", 
            "Makespan", "Runtime (s)", 
            "Compute Cost (₹)", "Solution Cost (₹)", "Total Cost (₹)"
        ])
        
        algorithms = {
            "Genetic Algorithm": OptimizationAlgorithms.genetic_algorithm,
            "Particle Swarm": OptimizationAlgorithms.particle_swarm,
            "Ant Colony": OptimizationAlgorithms.ant_colony,
            "Grey Wolf": OptimizationAlgorithms.grey_wolf,
            "Whale Optimization": OptimizationAlgorithms.whale_optimization,
            "Hybrid GWO-WOA": OptimizationAlgorithms.hybrid_grey_wolf_whale
        }
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        total_steps = len(selected_files) * len(algorithms)
        current_step = 0
        
        for file_path in selected_files:
            job_sizes = load_job_sizes(file_path)
            if not job_sizes:
                continue
            
            n_jobs = len(job_sizes)
            file_name = os.path.basename(file_path)
            
            for algo_name, algo_func in algorithms.items():
                status_text.text(f"Running {algo_name} on {file_name}...")
                
                makespan, runtime = algo_func(job_sizes, params[algo_name])
                compute_cost = runtime * compute_cost_per_sec
                solution_cost = makespan * solution_cost_per_unit
                total_cost = compute_cost + solution_cost
                
                new_row = pd.DataFrame({
                    "Dataset": [file_name],
                    "Jobs": [n_jobs],
                    "Algorithm": [algo_name],
                    "Makespan": [makespan],
                    "Runtime (s)": [runtime],
                    "Compute Cost (₹)": [compute_cost],
                    "Solution Cost (₹)": [solution_cost],
                    "Total Cost (₹)": [total_cost]
                })
                
                all_results = pd.concat([all_results, new_row], ignore_index=True)
                current_step += 1
                progress_bar.progress(current_step / total_steps)
        
        if all_results.empty:
            st.error("No valid results were generated. Please check your dataset files.")
            return
        
        # Display results
        st.subheader("📊 Results Summary")
        st.dataframe(
            all_results.style.format({
                "Runtime (s)": "{:.4f}",
                "Compute Cost (₹)": "₹{:.2f}",
                "Solution Cost (₹)": "₹{:.2f}",
                "Total Cost (₹)": "₹{:.2f}"
            }).highlight_min(subset=["Makespan", "Runtime (s)", "Total Cost (₹)"], color='lightgreen')
        )
        
        # Only show visualizations when running multiple files (By Dataset Size or All Dataset Sizes)
        if run_option in ["By Dataset Size", "All Dataset Sizes"]:
            st.subheader("📈 Performance Analysis")
            
            # Create tabs for different visualizations
            tab1, tab2, tab3 = st.tabs(["Solution Quality", "Runtime Performance", "Cost Analysis"])
            
            with tab1:
                # Apply abbreviations to algorithm names in visualization
                vis_df = all_results.copy()
                vis_df['Algorithm'] = vis_df['Algorithm'].map(lambda x: ALGO_ABBREVIATIONS.get(x, x))
                
                fig1 = px.line(vis_df, x="Jobs", y="Makespan", color="Algorithm",
                              title="<b>Algorithm Solution Quality by Problem Size</b>",
                              labels={"Jobs": "Number of Jobs", "Makespan": "Makespan"},
                              template="plotly_white")
                fig1.update_layout(hovermode="x unified", height=600)
                st.plotly_chart(fig1, use_container_width=True)
                
            with tab2:
                fig2 = px.line(vis_df, x="Jobs", y="Runtime (s)", color="Algorithm",
                              title="<b>Algorithm Runtime by Problem Size</b>",
                              labels={"Jobs": "Number of Jobs", "Runtime (s)": "Runtime (seconds)"},
                              template="plotly_white")
                fig2.update_layout(hovermode="x unified", height=600)
                st.plotly_chart(fig2, use_container_width=True)
                
            with tab3:
                fig3 = px.line(vis_df, x="Jobs", y="Total Cost (₹)", color="Algorithm",
                              title="<b>Algorithm Cost by Problem Size</b>",
                              labels={"Jobs": "Number of Jobs", "Total Cost (₹)": "Total Cost (₹)"},
                              template="plotly_white")
                fig3.update_layout(hovermode="x unified", height=600)
                st.plotly_chart(fig3, use_container_width=True)
                
                # Cost breakdown chart
                cost_breakdown = vis_df.melt(id_vars=["Dataset", "Jobs", "Algorithm"], 
                                           value_vars=["Compute Cost (₹)", "Solution Cost (₹)"],
                                           var_name="Cost Type", value_name="Cost (₹)")
                
                fig4 = px.bar(cost_breakdown, x="Algorithm", y="Cost (₹)", color="Cost Type",
                             title="<b>Cost Components by Algorithm</b>",
                             barmode="group", template="plotly_white")
                fig4.update_layout(height=600, xaxis_title="Algorithm", yaxis_title="Cost (₹)")
                st.plotly_chart(fig4, use_container_width=True)
        
        # Show best algorithm for each category
        st.subheader("🏆 Best Algorithm Recommendations by Dataset Size")

        # Categorize results
        all_results['Size Category'] = pd.cut(all_results['Jobs'],
                                            bins=[0, 300, 700, 1000],
                                            labels=['Small', 'Medium', 'Large'])

        # Create columns for each size category
        col1, col2, col3 = st.columns(3)

        with col1:
            small_df = all_results[all_results['Size Category'] == 'Small']
            if not small_df.empty:
                best_small = small_df.loc[small_df['Makespan'].idxmin()]
                st.metric(label="Best for Small Datasets (≤300 jobs)",
                         value=ALGO_ABBREVIATIONS.get(best_small['Algorithm'], best_small['Algorithm']),
                         delta=f"Makespan: {best_small['Makespan']:.2f}")
            else:
                st.warning("No small datasets in selection")

        with col2:
            medium_df = all_results[all_results['Size Category'] == 'Medium']
            if not medium_df.empty:
                best_medium = medium_df.loc[medium_df['Makespan'].idxmin()]
                st.metric(label="Best for Medium Datasets (300-700 jobs)",
                         value=ALGO_ABBREVIATIONS.get(best_medium['Algorithm'], best_medium['Algorithm']),
                         delta=f"Makespan: {best_medium['Makespan']:.2f}")
            else:
                st.warning("No medium datasets in selection")

        with col3:
            large_df = all_results[all_results['Size Category'] == 'Large']
            if not large_df.empty:
                best_large = large_df.loc[large_df['Makespan'].idxmin()]
                st.metric(label="Best for Large Datasets (>700 jobs)",
                         value=ALGO_ABBREVIATIONS.get(best_large['Algorithm'], best_large['Algorithm']),
                         delta=f"Makespan: {best_large['Makespan']:.2f}")
            else:
                st.warning("No large datasets in selection")

        # Add detailed comparison table
        st.subheader("📊 Detailed Performance by Dataset Size")

        # Calculate average metrics for each algorithm by size category
        avg_results = all_results.groupby(['Size Category', 'Algorithm']).agg({
            'Makespan': 'mean',
            'Runtime (s)': 'mean',
            'Total Cost (₹)': 'mean'
        }).reset_index()

        # Apply abbreviations to algorithm names
        avg_results['Algorithm'] = avg_results['Algorithm'].map(lambda x: ALGO_ABBREVIATIONS.get(x, x))

        # Pivot for better display
        pivot_results = avg_results.pivot(index='Algorithm', 
                                        columns='Size Category',
                                        values=['Makespan', 'Runtime (s)', 'Total Cost (₹)'])
        st.dataframe(
            pivot_results.style.format("{:.2f}")
            .highlight_min(axis=0, subset=[('Makespan', 'Small'), 
                                         ('Makespan', 'Medium'), 
                                         ('Makespan', 'Large')], 
                         color='lightgreen')
            .highlight_min(axis=0, subset=[('Runtime (s)', 'Small'), 
                                         ('Runtime (s)', 'Medium'), 
                                         ('Runtime (s)', 'Large')], 
                         color='lightblue')
            .highlight_min(axis=0, subset=[('Total Cost (₹)', 'Small'), 
                                         ('Total Cost (₹)', 'Medium'), 
                                         ('Total Cost (₹)', 'Large')], 
                         color='lightyellow')
        )

        # Aggregate Trends by Dataset Size
        st.subheader("📈 Aggregate Trends by Dataset Size")

        # Create a faceted plot showing performance by size category
        fig5 = px.bar(avg_results, 
                     x='Algorithm', 
                     y='Makespan',
                     facet_col='Size Category',
                     title="<b>Average Makespan by Algorithm and Dataset Size</b>",
                     template="plotly_white")
        st.plotly_chart(fig5, use_container_width=True)

        fig6 = px.bar(avg_results, 
                     x='Algorithm', 
                     y='Runtime (s)',
                     facet_col='Size Category',
                     title="<b>Average Runtime by Algorithm and Dataset Size</b>",
                     template="plotly_white")
        st.plotly_chart(fig6, use_container_width=True)
        
        # Add statistical analysis
        perform_statistical_analysis(all_results)
        
        # Add algorithm behavior analysis
        plot_algorithm_behavior(all_results)
        
        # Add download button for results
        st.subheader("📥 Download Results")
        csv = all_results.to_csv(index=False)
        st.download_button(
            label="Download results as CSV",
            data=csv,
            file_name='algorithm_comparison_results.csv',
            mime='text/csv'
        )

if __name__ == "__main__":
    main()