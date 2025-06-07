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
        """Calculate fitness of an individual"""
        total_weight = sum(individual[i] * self.items[i][0] for i in range(self.num_items))
        total_value = sum(individual[i] * self.items[i][1] for i in range(self.num_items))
        
        # If solution exceeds capacity, it's invalid (fitness = 0)
        if total_weight > self.capacity:
            return 0
        
        return total_value
    
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
                
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                
                new_population.extend([child1, child2])
            
            # Ensure population size is exact
            population = new_population[:self.population_size]
            
            # Print progress every 50 generations
            if generation % 50 == 0 or generation == self.generations - 1:
                print(f"Generation {generation}: Best Fitness = {best_gen_fitness}, Avg Fitness = {avg_gen_fitness:.2f}")
        
        return self.get_solution_details()
    
    def get_solution_details(self) -> Dict:
        """Get detailed information about the best solution"""
        if self.best_solution is None:
            return {'total_value': 0, 'total_weight': 0, 'selected_items': [], 'capacity_used': 0}
        
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
            'capacity_used': total_weight / self.capacity * 100 if self.capacity > 0 else 0
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

def simple_test():
    """Simple test to verify the algorithm works"""
    print("="*50)
    print("ALGORITHM VERIFICATION TEST")
    print("="*50)
    
    # Simple test case
    items = [(10, 60), (20, 100), (30, 120)]
    capacity = 50
    
    print(f"Test Problem:")
    print(f"Items: {items}")
    print(f"Capacity: {capacity}")
    
    # Get optimal solution
    optimal_value = dynamic_programming_solution(items, capacity)
    print(f"Optimal Solution: {optimal_value}")
    
    # Test GA approach
    print(f"\n--- Genetic Algorithm Solution ---")
    ga = KnapsackGA(items=items, capacity=capacity, 
                   population_size=50, generations=100)
    solution = ga.evolve()
    
    # Show results
    print(f"\n=== RESULTS ===")
    print(f"Optimal Solution: {optimal_value}")
    print(f"GA Solution: {solution['total_value']} ({solution['total_value']/optimal_value*100:.1f}% of optimal)")
    
    if solution['total_value']/optimal_value >= 0.8:
        print("✓ Test PASSED - GA found good solution")
    else:
        print("✗ Test FAILED - GA solution quality too low")
    
    # Show evolution
    ga.plot_evolution()

if __name__ == "__main__":
    # Run verification test
    simple_test()
    
    print("\n" + "="*60)
    print("MAIN EXAMPLE - CLASSICAL KNAPSACK PROBLEM")
    print("="*60)
    
    # Classical knapsack problem
    items = [
        (10, 60),   # Item 1: Weight=10, Value=60
        (20, 100),  # Item 2: Weight=20, Value=100  
        (30, 120),  # Item 3: Weight=30, Value=120
        (5, 30),    # Item 4: Weight=5,  Value=30
        (15, 75),   # Item 5: Weight=15, Value=75
    ]
    
    capacity = 50
    
    print(f"Knapsack Capacity: {capacity}")
    print(f"Available Items:")
    for i, (weight, value) in enumerate(items, 1):
        print(f"  Item {i}: Weight={weight}, Value={value}, Density={value/weight:.2f}")
    
    # Solve using Genetic Algorithm
    print(f"\n--- SOLVING WITH GENETIC ALGORITHM ---")
    
    ga = KnapsackGA(
        items=items,
        capacity=capacity,
        population_size=100,
        generations=150,
        mutation_rate=0.02,
        crossover_rate=0.8
    )
    
    solution = ga.evolve()
    
    # Display results
    print(f"\n=== FINAL SOLUTION ===")
    print(f"Total Value: {solution['total_value']}")
    print(f"Total Weight: {solution['total_weight']}/{capacity}")
    print(f"Capacity Used: {solution['capacity_used']:.1f}%")
    
    print(f"\nSelected Items:")
    for item in solution['selected_items']:
        print(f"  Item {item['item']}: Weight={item['weight']}, Value={item['value']}")
    
    # Compare with optimal
    optimal_value = dynamic_programming_solution(items, capacity)
    print(f"\nOptimal Solution: {optimal_value}")
    print(f"GA Solution Quality: {solution['total_value']/optimal_value*100:.1f}% of optimal")
    
    # Show evolution plot
    ga.plot_evolution()