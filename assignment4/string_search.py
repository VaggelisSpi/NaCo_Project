import random
from typing import List, Callable, Tuple

def string_genetic_algorithm(
    alphabet: List[str],
    target: str,
    string_length: int = 5,
    population_size: int = 1,
    mu: float = 0.05,
    tournament_size: int = 1,
    max_generations: int = 100,
    debug: bool = False,
) -> Tuple[List[List[str]], List[float], int]:
    # Fitness function: number of matches with target
    def fitness(individual):
        return sum(1 for a, b in zip(individual, target) if a == b)

    # Initialize population
    population = []
    for _ in range(population_size):
        individual = [random.choice(alphabet) for _ in range(string_length)]
        population.append(individual)    

    # List to store average fitness per generation
    avg_fitness = []

    for generation in range(max_generations):
        # List to store new generation
        new_population: List[List[str]] = []

        # Number of tournaments (assuming no elitism)
        # Since we need to create N children, with each tournament creating one parent
        # For generational replacement, all new_population is filled with children
        while len(new_population) < population_size:
            # Tournament selection for parent 1
            tournament = random.sample(population, tournament_size)
            parent1 = max(tournament, key=lambda x: fitness(x))

            # Tournament selection for parent 2
            tournament2 = random.sample(population, tournament_size)
            parent2 = max(tournament2, key=lambda x: fitness(x))

            # Random crossover point
            crossover_point = random.randint(1, len(parent1)-1)
            child1 = parent1[:crossover_point] + parent2[crossover_point:]
            child2 = parent2[:crossover_point] + parent1[crossover_point:]

            # Apply mutation to both children
            def mutate(child):
                mutated = []
                for gene in child:
                    if random.random() < mu:
                        mutated.append(random.choice(alphabet))
                    else:
                        mutated.append(gene)
                return mutated

            child1 = mutate(child1)
            child2 = mutate(child2)

            # Add children to new_population
            new_population.append(child1)
            new_population.append(child2)

        # Replace population with new_population
        population = new_population

        # Log average fitness
        current_fitness = [fitness(individual) for individual in population]
        avg_fit = sum(current_fitness) / len(current_fitness)
        avg_fitness.append(avg_fit)

        # Check if target is reached
        if any(''.join(ind) == target for ind in population):
            print(f"Target reached in {generation + 1} generations!")
            break

        if generation > max_generations:
            print("Max generations reached without finding the solution!")
            break

        if debug:
            # Print progress (optional)
            if generation % 10 == 0:
                print(f"Generation {generation}: Average Fitness = {avg_fit:.2f}")
    
    return population, avg_fitness, generation


if __name__ == "__main__":
    english_letters = [
        'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 
        'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 
        'u', 'v', 'w', 'x', 'y', 'z', 
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 
        'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 
        'U', 'V', 'W', 'X', 'Y', 'Z'
    ]
    res, fits, generation = string_genetic_algorithm(english_letters, "abcdefghijklmno", 15, 200, 0.06, 2)