# has Death Penalty vs. Soft Penalty 

import random
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict

class KnapsackGA:
    def __init__(self, items: List[Tuple[int, int]], capacity: int, 
                 population_size: int = 100, generations: int = 200,
                 mutation_rate: float = 0.01, crossover_rate: float = 0.8):
        """
        Initialize Genetic Algorithm for 0/1 Knapsack Problem
        
        Args:
            items: List of (weight, value) tuples
            capacity: Maximum weight capacity of knapsack
            population_size: Size of population in each generation
            generations: Number of generations to evolve
            mutation_rate: Probability of mutation for each gene
            crossover_rate: Probability of crossover between parents
        """
        self.items = items
        self.capacity = capacity
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.num_items = len(items)
        
        # Track evolution statistics
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.best_solution = None
        self.best_fitness = 0
    
    def create_individual(self) -> List[int]:
        """Create a random individual (chromosome) as binary list"""
        return [random.randint(0, 1) for _ in range(self.num_items)]
    
    def create_population(self) -> List[List[int]]:
        """Create initial population of random individuals"""
        return [self.create_individual() for _ in range(self.population_size)]
    
    def calculate_fitness(self, individual: List[int]) -> float:
        """
        Calculate fitness of an individual
        Uses penalty method instead of death penalty for smoother evolution
        """
        total_weight = sum(individual[i] * self.items[i][0] for i in range(self.num_items))
        total_value = sum(individual[i] * self.items[i][1] for i in range(self.num_items))
        
        # Penalty for exceeding capacity (softer penalty)
        if total_weight > self.capacity:
            # Return penalized fitness instead of 0
            penalty = (total_weight - self.capacity) * 10  # Penalty factor
            return max(0, total_value - penalty)
        
        return total_value
    
    def repair_solution(self, individual: List[int]) -> List[int]:
        """
        Repair infeasible solutions by removing items with lowest value density
        """
        if not individual or sum(individual) == 0:
            return individual
        
        # Calculate current weight
        total_weight = sum(individual[i] * self.items[i][0] for i in range(self.num_items))
        
        if total_weight <= self.capacity:
            return individual  # Already feasible
        
        # Create list of selected items with their indices and value density
        selected_items = []
        for i in range(self.num_items):
            if individual[i] == 1:
                weight, value = self.items[i]
                density = value / weight if weight > 0 else 0
                selected_items.append((i, weight, value, density))
        
        # Sort by value density (ascending) to remove least efficient items first
        selected_items.sort(key=lambda x: x[3])
        
        # Remove items until feasible
        repaired = individual[:]
        current_weight = total_weight
        
        for item_idx, weight, value, density in selected_items:
            if current_weight <= self.capacity:
                break
            repaired[item_idx] = 0
            current_weight -= weight
        
        return repaired
    
    def tournament_selection(self, population: List[List[int]], tournament_size: int = 3) -> List[int]:
        """Select parent using tournament selection"""
        tournament = random.sample(population, tournament_size)
        return max(tournament, key=self.calculate_fitness)
    
    def crossover(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        """Single-point crossover between two parents"""
        if random.random() > self.crossover_rate or self.num_items <= 1:
            return parent1[:], parent2[:]
        
        crossover_point = random.randint(1, self.num_items - 1)
        
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        
        return child1, child2
    
    def mutate(self, individual: List[int]) -> List[int]:
        """Bit-flip mutation"""
        mutated = individual[:]
        for i in range(self.num_items):
            if random.random() < self.mutation_rate:
                mutated[i] = 1 - mutated[i]  # Flip bit
        return mutated
    
    def evolve(self) -> Dict:
        """Main evolution loop"""
        population = self.create_population()
        
        for generation in range(self.generations):
            # Calculate fitness for all individuals
            fitness_scores = [self.calculate_fitness(ind) for ind in population]
            
            # Track statistics
            best_gen_fitness = max(fitness_scores)
            avg_gen_fitness = sum(fitness_scores) / len(fitness_scores)
            
            self.best_fitness_history.append(best_gen_fitness)
            self.avg_fitness_history.append(avg_gen_fitness)
            
            # Update best solution
            if best_gen_fitness > self.best_fitness:
                self.best_fitness = best_gen_fitness
                best_idx = fitness_scores.index(best_gen_fitness)
                self.best_solution = population[best_idx][:]
            
            # Create new population
            new_population = []
            
            # Elitism: Keep best individual
            best_idx = fitness_scores.index(max(fitness_scores))
            new_population.append(population[best_idx][:])
            
            # Generate rest of population
            while len(new_population) < self.population_size:
                parent1 = self.tournament_selection(population)
                parent2 = self.tournament_selection(population)
                
                child1, child2 = self.crossover(parent1, parent2)
                
                # Repair infeasible solutions
                child1 = self.repair_solution(child1)
                child2 = self.repair_solution(child2)
                
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                
                # Repair again after mutation
                child1 = self.repair_solution(child1)
                child2 = self.repair_solution(child2)
                
                new_population.extend([child1, child2])
            
            # Ensure population size is exact
            population = new_population[:self.population_size]
            
            # Print progress
            if generation % 20 == 0 or generation == self.generations - 1:
                print(f"Generation {generation}: Best Fitness = {best_gen_fitness}, Avg Fitness = {avg_gen_fitness:.2f}")
        
        # Return solution details
        return self.get_solution_details()
    
    def get_solution_details(self) -> Dict:
        """Get detailed information about the best solution"""
        if self.best_solution is None:
            return {'total_value': 0, 'total_weight': 0, 'selected_items': [], 'capacity_used': 0, 'num_items_selected': 0}
        
        selected_items = []
        total_weight = 0
        total_value = 0
        
        for i, selected in enumerate(self.best_solution):
            if selected:
                weight, value = self.items[i]
                selected_items.append({
                    'item': i + 1,
                    'weight': weight,
                    'value': value,
                    'value_density': value / weight
                })
                total_weight += weight
                total_value += value
        
        return {
            'solution': self.best_solution,
            'selected_items': selected_items,
            'total_weight': total_weight,
            'total_value': total_value,
            'capacity_used': total_weight / self.capacity * 100,
            'num_items_selected': len(selected_items)
        }
    
    def plot_evolution(self):
        """Plot the evolution of fitness over generations"""
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.best_fitness_history, 'r-', label='Best Fitness', linewidth=2)
        plt.plot(self.avg_fitness_history, 'b-', label='Average Fitness', linewidth=2)
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.title('Evolution of Fitness')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(self.best_fitness_history, 'g-', linewidth=2)
        plt.xlabel('Generation')
        plt.ylabel('Best Fitness')
        plt.title('Best Fitness Progress')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

def dynamic_programming_solution(items: List[Tuple[int, int]], capacity: int) -> int:
    """Optimal solution using Dynamic Programming for comparison"""
    n = len(items)
    dp = [[0 for _ in range(capacity + 1)] for _ in range(n + 1)]
    
    for i in range(1, n + 1):
        weight, value = items[i-1]
        for w in range(capacity + 1):
            if weight <= w:
                dp[i][w] = max(dp[i-1][w], dp[i-1][w-weight] + value)
            else:
                dp[i][w] = dp[i-1][w]
    
    return dp[n][capacity]

def run_test_cases():
    """Run essential test cases to validate the algorithm"""
    print("="*60)
    print("RUNNING TEST CASES")
    print("="*60)
    
    test_cases = [
        {
            "name": "Basic Test",
            "items": [(2, 3), (3, 4), (4, 5), (5, 6)],
            "capacity": 5,
            "description": "Simple 4-item problem"
        },
        {
            "name": "Classical Example", 
            "items": [(10, 60), (20, 100), (30, 120)],
            "capacity": 50,
            "description": "Standard textbook problem"
        },
        {
            "name": "Edge Case - Single Item",
            "items": [(5, 100)],
            "capacity": 10,
            "description": "Only one item that fits"
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test['name']}")
        print(f"Items: {test['items']}, Capacity: {test['capacity']}")
        
        # Get optimal solution
        optimal_value = dynamic_programming_solution(test['items'], test['capacity'])
        
        # Test with GA
        ga = KnapsackGA(items=test['items'], capacity=test['capacity'], 
                       population_size=50, generations=50)
        solution = ga.evolve()
        ga_value = solution.get('total_value', 0)
        quality = (ga_value / optimal_value * 100) if optimal_value > 0 else 100
        
        print(f"Optimal: {optimal_value}, GA: {ga_value}, Quality: {quality:.1f}%")
        print(f"Result: {'PASS' if quality >= 80 else 'FAIL'}")
    
    return True

def performance_benchmark():
    """Benchmark the algorithm on different problem sizes"""
    print("\n" + "="*60)
    print("PERFORMANCE BENCHMARK")
    print("="*60)
    
    problem_sizes = [10, 20, 50, 100]
    
    for size in problem_sizes:
        print(f"\nProblem Size: {size} items")
        
        # Generate random problem
        random.seed(42)  # For reproducible results
        items = [(random.randint(1, 50), random.randint(10, 100)) for _ in range(size)]
        capacity = sum(item[0] for item in items) // 3  # About 1/3 of total weight
        
        # Time the GA
        import time
        start_time = time.time()
        
        ga = KnapsackGA(
            items=items,
            capacity=capacity,
            population_size=min(100, size * 2),
            generations=min(200, size * 4),
            mutation_rate=0.02,
            crossover_rate=0.8
        )
        
        solution = ga.evolve()
        end_time = time.time()
        
        # Get optimal for smaller problems
        if size <= 20:
            optimal_value = dynamic_programming_solution(items, capacity)
            quality = solution['total_value'] / optimal_value * 100
            print(f"Optimal Value: {optimal_value}")
            print(f"GA Quality: {quality:.1f}%")
        
        print(f"GA Value: {solution['total_value']}")
        print(f"Execution Time: {end_time - start_time:.2f} seconds")
        print(f"Items Selected: {solution['num_items_selected']}/{size}")
        print(f"Capacity Used: {solution['capacity_used']:.1f}%")

def validate_solution_feasibility():
    """Test that all solutions are feasible"""
    print("\n" + "="*60)
    print("SOLUTION FEASIBILITY VALIDATION")
    print("="*60)
    
    # Test multiple runs to check consistency
    items = [(10, 60), (20, 100), (30, 120), (5, 30), (15, 75)]
    capacity = 50
    
    feasible_count = 0
    total_runs = 10
    
    for run in range(total_runs):
        ga = KnapsackGA(items=items, capacity=capacity, population_size=50, generations=100)
        solution = ga.evolve()
        
        # Check feasibility
        total_weight = sum(solution['solution'][i] * items[i][0] for i in range(len(items)))
        is_feasible = total_weight <= capacity
        
        if is_feasible:
            feasible_count += 1
        
        print(f"Run {run+1}: Weight={total_weight}/{capacity}, Value={solution['total_value']}, Feasible={is_feasible}")
    
    print(f"\nFeasibility Rate: {feasible_count}/{total_runs} ({feasible_count/total_runs*100:.1f}%)")
    return feasible_count == total_runs

# Example usage and testing
def compare_fitness_approaches():
    """Compare death penalty vs. repair method approaches"""
    print("\n" + "="*60)
    print("COMPARING FITNESS APPROACHES")
    print("="*60)
    
    items = [(10, 60), (20, 100), (30, 120), (5, 30), (15, 75)]
    capacity = 50
    
    # Death penalty approach (original)
    class DeathPenaltyGA(KnapsackGA):
        def calculate_fitness(self, individual):
            total_weight = sum(individual[i] * self.items[i][0] for i in range(self.num_items))
            total_value = sum(individual[i] * self.items[i][1] for i in range(self.num_items))
            return total_value if total_weight <= self.capacity else 0
        
        def repair_solution(self, individual):
            return individual  # No repair
    
    print("1. Death Penalty Approach (Original):")
    ga_death = DeathPenaltyGA(items=items, capacity=capacity, population_size=100, generations=150)
    solution_death = ga_death.evolve()
    
    print("2. Repair Method Approach (Improved):")
    ga_repair = KnapsackGA(items=items, capacity=capacity, population_size=100, generations=150)
    solution_repair = ga_repair.evolve()
    
    # Plot comparison
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(ga_death.best_fitness_history, 'r-', label='Death Penalty - Best', linewidth=2)
    plt.plot(ga_death.avg_fitness_history, 'r--', label='Death Penalty - Avg', alpha=0.7)
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Death Penalty Approach')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.plot(ga_repair.best_fitness_history, 'b-', label='Repair Method - Best', linewidth=2)
    plt.plot(ga_repair.avg_fitness_history, 'b--', label='Repair Method - Avg', alpha=0.7)
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Repair Method Approach')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    plt.plot(ga_death.avg_fitness_history, 'r--', label='Death Penalty - Avg', alpha=0.7)
    plt.plot(ga_repair.avg_fitness_history, 'b--', label='Repair Method - Avg', alpha=0.7)
    plt.xlabel('Generation')
    plt.ylabel('Average Fitness')
    plt.title('Average Fitness Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nResults Comparison:")
    print(f"Death Penalty - Final Value: {solution_death['total_value']}")
    print(f"Repair Method - Final Value: {solution_repair['total_value']}")
    
    # Calculate average fitness variance
    death_variance = np.var(ga_death.avg_fitness_history)
    repair_variance = np.var(ga_repair.avg_fitness_history)
    
    print(f"Death Penalty - Avg Fitness Variance: {death_variance:.2f}")
    print(f"Repair Method - Avg Fitness Variance: {repair_variance:.2f}")
    print(f"Smoothness Improvement: {((death_variance - repair_variance) / death_variance * 100):.1f}%")

if __name__ == "__main__":
    # Run basic tests
    print("Running basic validation tests...")
    run_test_cases()
    
    # Main example problem
    print("\n" + "="*60)
    print("MAIN EXAMPLE PROBLEM")
    print("="*60)
    
    # Example problem: Classical knapsack items (weight, value)
    items = [
        (10, 60),   # Item 1
        (20, 100),  # Item 2
        (30, 120),  # Item 3
        (5, 30),    # Item 4
        (15, 75),   # Item 5
        (25, 90),   # Item 6
        (12, 45),   # Item 7
        (8, 40),    # Item 8
        (18, 85),   # Item 9
        (22, 110)   # Item 10
    ]
    
    capacity = 50
    
    print("="*60)
    print("0/1 KNAPSACK PROBLEM - GENETIC ALGORITHM SOLUTION")
    print("="*60)
    print(f"Knapsack Capacity: {capacity}")
    print(f"Number of Items: {len(items)}")
    print("\nItems (Weight, Value):")
    for i, (w, v) in enumerate(items, 1):
        print(f"Item {i}: Weight={w}, Value={v}, Density={v/w:.2f}")
    
    # Solve using Genetic Algorithm
    print("\n" + "="*40)
    print("GENETIC ALGORITHM SOLUTION")
    print("="*40)
    
    ga = KnapsackGA(
        items=items,
        capacity=capacity,
        population_size=100,
        generations=200,
        mutation_rate=0.02,
        crossover_rate=0.8
    )
    
    solution = ga.evolve()
    
    print(f"\nBest Solution Found:")
    print(f"Total Value: {solution['total_value']}")
    print(f"Total Weight: {solution['total_weight']}/{capacity}")
    print(f"Capacity Used: {solution['capacity_used']:.1f}%")
    print(f"Items Selected: {solution['num_items_selected']}")
    
    print(f"\nSelected Items:")
    for item in solution['selected_items']:
        print(f"Item {item['item']}: Weight={item['weight']}, Value={item['value']}, Density={item['value_density']:.2f}")
    
    # Compare with optimal solution
    optimal_value = dynamic_programming_solution(items, capacity)
    print(f"\nOptimal Solution (Dynamic Programming): {optimal_value}")
    print(f"GA Solution Quality: {solution['total_value']/optimal_value*100:.1f}% of optimal")
    
    # Plot evolution
    ga.plot_evolution()
    
    # Test with different parameters
    print("\n" + "="*40)
    print("PARAMETER SENSITIVITY ANALYSIS")
    print("="*40)
    
    configs = [
        {"population_size": 50, "generations": 100, "mutation_rate": 0.01},
        {"population_size": 100, "generations": 200, "mutation_rate": 0.02},
        {"population_size": 200, "generations": 300, "mutation_rate": 0.03}
    ]
    
    for i, config in enumerate(configs, 1):
        print(f"\nConfiguration {i}: {config}")
        ga_test = KnapsackGA(items=items, capacity=capacity, **config)
        result = ga_test.evolve()
        quality = result['total_value'] / optimal_value * 100
        print(f"Result: Value={result['total_value']}, Quality={quality:.1f}%")