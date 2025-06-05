import numpy as np
import matplotlib.pyplot as plt
import random
from typing import List, Tuple
import time

class TSPGeneticAlgorithm:
    def __init__(self, cities: np.ndarray, population_size: int = 100, 
                 elite_size: int = 20, mutation_rate: float = 0.01, generations: int = 500):
        """
        Initialize TSP Genetic Algorithm
        
        Args:
            cities: Array of city coordinates [(x1,y1), (x2,y2), ...]
            population_size: Size of population in each generation
            elite_size: Number of best routes to keep in each generation
            mutation_rate: Probability of mutation
            generations: Number of generations to evolve
        """
        self.cities = cities
        self.num_cities = len(cities)
        self.population_size = population_size
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate
        self.generations = generations
        self.distance_matrix = self._calculate_distance_matrix()
        
    def _calculate_distance_matrix(self) -> np.ndarray:
        """Calculate distance matrix between all cities"""
        n = self.num_cities
        dist_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    dist_matrix[i][j] = np.sqrt(
                        (self.cities[i][0] - self.cities[j][0])**2 + 
                        (self.cities[i][1] - self.cities[j][1])**2
                    )
        return dist_matrix
    
    def _create_route(self) -> List[int]:
        """Create a random route visiting all cities"""
        route = list(range(self.num_cities))
        random.shuffle(route)
        return route
    
    def _calculate_fitness(self, route: List[int]) -> float:
        """Calculate fitness (inverse of total distance) for a route"""
        total_distance = 0
        for i in range(len(route)):
            from_city = route[i]
            to_city = route[(i + 1) % len(route)]  # Return to start city
            total_distance += self.distance_matrix[from_city][to_city]
        
        # Fitness is inverse of distance (higher fitness = shorter distance)
        return 1 / total_distance if total_distance > 0 else float('inf')
    
    def _calculate_distance(self, route: List[int]) -> float:
        """Calculate total distance for a route"""
        return 1 / self._calculate_fitness(route)
    
    def _create_initial_population(self) -> List[List[int]]:
        """Create initial population of random routes"""
        population = []
        for _ in range(self.population_size):
            population.append(self._create_route())
        return population
    
    def _rank_routes(self, population: List[List[int]]) -> List[Tuple[int, float]]:
        """Rank routes by fitness"""
        fitness_results = []
        for i, route in enumerate(population):
            fitness = self._calculate_fitness(route)
            fitness_results.append((i, fitness))
        
        # Sort by fitness (descending - higher fitness is better)
        return sorted(fitness_results, key=lambda x: x[1], reverse=True)
    
    def _selection(self, ranked_pop: List[Tuple[int, float]]) -> List[int]:
        """Select parents for breeding using tournament selection"""
        selection_results = []
        
        # Keep elite routes
        for i in range(self.elite_size):
            selection_results.append(ranked_pop[i][0])
        
        # Select remaining parents using roulette wheel selection
        fitness_sum = sum([fitness for _, fitness in ranked_pop])
        
        for _ in range(len(ranked_pop) - self.elite_size):
            pick = random.uniform(0, fitness_sum)
            current = 0
            for i, (route_idx, fitness) in enumerate(ranked_pop):
                current += fitness
                if current >= pick:
                    selection_results.append(route_idx)
                    break
        
        return selection_results
    
    def _breed(self, parent1: List[int], parent2: List[int]) -> List[int]:
        """Create offspring using ordered crossover (OX)"""
        start = random.randint(0, len(parent1) - 1)
        end = random.randint(start + 1, len(parent1))
        
        # Copy segment from parent1
        child = [-1] * len(parent1)
        child[start:end] = parent1[start:end]
        
        # Fill remaining positions with parent2's order
        remaining = [city for city in parent2 if city not in child]
        j = 0
        for i in range(len(child)):
            if child[i] == -1:
                child[i] = remaining[j]
                j += 1
        
        return child
    
    def _breed_population(self, mating_pool: List[List[int]]) -> List[List[int]]:
        """Create new population through breeding"""
        children = []
        
        # Keep elite routes
        for i in range(self.elite_size):
            children.append(mating_pool[i])
        
        # Breed remaining population
        for _ in range(len(mating_pool) - self.elite_size):
            parent1 = random.choice(mating_pool[:50])  # Bias towards better parents
            parent2 = random.choice(mating_pool[:50])
            child = self._breed(parent1, parent2)
            children.append(child)
        
        return children
    
    def _mutate(self, individual: List[int]) -> List[int]:
        """Mutate individual using swap mutation"""
        if random.random() < self.mutation_rate:
            # Swap two random cities
            i, j = random.sample(range(len(individual)), 2)
            individual[i], individual[j] = individual[j], individual[i]
        return individual
    
    def _mutate_population(self, population: List[List[int]]) -> List[List[int]]:
        """Apply mutation to population"""
        mutated_pop = []
        for individual in population:
            mutated_pop.append(self._mutate(individual[:]))  # Copy to avoid modifying original
        return mutated_pop
    
    def solve(self) -> Tuple[List[int], float, List[float]]:
        """
        Solve TSP using Genetic Algorithm
        
        Returns:
            best_route: Best route found
            best_distance: Distance of best route
            progress: List of best distances per generation
        """
        print("Initializing population...")
        population = self._create_initial_population()
        progress = []
        
        print(f"Evolving for {self.generations} generations...")
        for generation in range(self.generations):
            # Rank current population
            ranked_pop = self._rank_routes(population)
            
            # Track progress
            best_distance = 1 / ranked_pop[0][1]
            progress.append(best_distance)
            
            if generation % 50 == 0:
                print(f"Generation {generation}: Best distance = {best_distance:.2f}")
            
            # Select parents
            selection_results = self._selection(ranked_pop)
            mating_pool = [population[i] for i in selection_results]
            
            # Breed new population
            children = self._breed_population(mating_pool)
            
            # Mutate population
            population = self._mutate_population(children)
        
        # Get final best route
        final_ranked = self._rank_routes(population)
        best_route_index = final_ranked[0][0]
        best_route = population[best_route_index]
        best_distance = 1 / final_ranked[0][1]
        
        return best_route, best_distance, progress
    
    def visualize_route(self, route: List[int], title: str = "TSP Route"):
        """Visualize the route on a plot"""
        plt.figure(figsize=(10, 8))
        
        # Plot cities
        x_coords = [self.cities[i][0] for i in route] + [self.cities[route[0]][0]]
        y_coords = [self.cities[i][1] for i in route] + [self.cities[route[0]][1]]
        
        plt.plot(x_coords, y_coords, 'bo-', linewidth=2, markersize=8)
        plt.plot(x_coords[0], y_coords[0], 'ro', markersize=12, label='Start/End')
        
        # Add city labels
        for i, city_idx in enumerate(route):
            plt.annotate(f'{city_idx}', 
                        (self.cities[city_idx][0], self.cities[city_idx][1]),
                        xytext=(5, 5), textcoords='offset points')
        
        plt.title(title)
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def plot_progress(self, progress: List[float]):
        """Plot the evolution progress"""
        plt.figure(figsize=(10, 6))
        plt.plot(progress, 'b-', linewidth=2)
        plt.title('Genetic Algorithm Progress')
        plt.xlabel('Generation')
        plt.ylabel('Best Distance')
        plt.grid(True, alpha=0.3)
        plt.show()

def create_random_cities(num_cities: int, width: int = 100, height: int = 100) -> np.ndarray:
    """Create random city coordinates"""
    return np.random.rand(num_cities, 2) * [width, height]

def create_circle_cities(num_cities: int, radius: int = 50) -> np.ndarray:
    """Create cities arranged in a circle (for testing)"""
    angles = np.linspace(0, 2*np.pi, num_cities, endpoint=False)
    cities = np.array([[radius * np.cos(angle) + 50, radius * np.sin(angle) + 50] 
                      for angle in angles])
    return cities

# Example usage and demonstration
if __name__ == "__main__":
    # Example 1: Random cities
    print("=== TSP Genetic Algorithm ===")
    print("\n1. Solving TSP with random cities...")
    
    # Create random cities
    num_cities = 15
    cities = create_random_cities(num_cities, 100, 100)
    
    # Set up and solve TSP
    tsp_ga = TSPGeneticAlgorithm(
        cities=cities,
        population_size=100,
        elite_size=20,
        mutation_rate=0.01,
        generations=300
    )
    
    start_time = time.time()
    best_route, best_distance, progress = tsp_ga.solve()
    end_time = time.time()
    
    print(f"\nSolution found in {end_time - start_time:.2f} seconds")
    print(f"Best route: {best_route}")
    print(f"Best distance: {best_distance:.2f}")
    print(f"Improvement: {progress[0]:.2f} -> {progress[-1]:.2f}")
    
    # Visualize results
    tsp_ga.visualize_route(best_route, f"Best TSP Route (Distance: {best_distance:.2f})")
    tsp_ga.plot_progress(progress)
    
    # Example 2: Circle cities (optimal solution known)
    print("\n2. Testing with cities in a circle...")
    circle_cities = create_circle_cities(10)
    
    tsp_circle = TSPGeneticAlgorithm(
        cities=circle_cities,
        population_size=50,
        elite_size=10,
        mutation_rate=0.02,
        generations=200
    )
    
    circle_route, circle_distance, circle_progress = tsp_circle.solve()
    
    print(f"Circle TSP - Best distance: {circle_distance:.2f}")
    tsp_circle.visualize_route(circle_route, "Circle Cities TSP")
    
    # Compare with simple nearest neighbor heuristic
    def nearest_neighbor_tsp(cities):
        """Simple nearest neighbor heuristic for comparison"""
        n = len(cities)
        unvisited = set(range(1, n))
        route = [0]
        current = 0
        
        while unvisited:
            nearest = min(unvisited, 
                         key=lambda x: np.sqrt((cities[current][0] - cities[x][0])**2 + 
                                             (cities[current][1] - cities[x][1])**2))
            route.append(nearest)
            unvisited.remove(nearest)
            current = nearest
            
        return route
    
    nn_route = nearest_neighbor_tsp(cities)
    nn_distance = tsp_ga._calculate_distance(nn_route)
    
    print(f"\nComparison:")
    print(f"Genetic Algorithm: {best_distance:.2f}")
    print(f"Nearest Neighbor:  {nn_distance:.2f}")
    print(f"GA Improvement:    {((nn_distance - best_distance) / nn_distance * 100):.1f}%")