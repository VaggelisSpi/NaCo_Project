import random
from typing import List, Callable, Tuple


class GeneticAlgorithm:
    def __init__(
        self,
        alphabet: List[str],
        target: str,
        string_length: int = 5,
        population_size: int = 1,
        mu: float = 0.05,
        tournament_size: int = 1,
        max_generations: int = 100,
        generations_to_analyse: int = 10,
    ) -> None:
        self.alphabet = alphabet
        self.target = target
        self.string_length = string_length
        self.population_size = population_size
        self.mu = mu
        self.tournament_size = tournament_size
        self.max_generations = max_generations
        self.generations_to_analyse = generations_to_analyse
        # List to store average fitness per generation
        self.avg_fitness: List[float] = []
        self.population: List[List[str]] = []
        self.hamming_distances: List[float] = []

    def fitness(self, individual):
        # fitness function: number of matches with target
        return sum(1 for a, b in zip(individual, self.target) if a == b)

    def get_avg_fitness(self) -> List[float]:
        return self.avg_fitness

    def get_hamming_distances(self) -> List[float]:
        return self.hamming_distances

    @staticmethod
    def hamming_distance(string1, string2):
        return sum(char1 != char2 for char1, char2 in zip(string1, string2))

    def analyse_diversity(self) -> None:
        """
        Calculate diversity metrics
        For a random sample of the population calculate the hamming distances between them and get the avg

        This will be run every few generations
        """
        sample_size = 50
        if len(self.population) < sample_size:
            sample = self.population
        else:
            sample = random.sample(self.population, sample_size)

        distances = []
        for i in range(len(sample)):
            for j in range(i + 1, len(sample)):
                distances.append(self.hamming_distance(sample[i], sample[j]))

        if not distances:
            self.hamming_distances.append(0.0)
        else:
            self.hamming_distances.append(sum(distances) / len(distances))

    def run(self) -> Tuple[List[List[str]], int]:
        # Initialize population
        for _ in range(self.population_size):
            individual = [random.choice(self.alphabet) for _ in range(self.string_length)]
            self.population.append(individual)

        for generation in range(self.max_generations):
            # List to store new generation
            new_population: List[List[str]] = []

            # Number of tournaments (assuming no elitism)
            # Since we need to create N children, with each tournament creating one parent
            # For generational replacement, all new_population is filled with children
            while len(new_population) < self.population_size:
                # Tournament selection for parent 1
                tournament = random.sample(self.population, self.tournament_size)
                parent1 = max(tournament, key=lambda x: self.fitness(x))

                # Tournament selection for parent 2
                tournament2 = random.sample(self.population, self.tournament_size)
                parent2 = max(tournament2, key=lambda x: self.fitness(x))

                # Random crossover point
                crossover_point = random.randint(1, len(parent1) - 1)
                child1 = parent1[:crossover_point] + parent2[crossover_point:]
                child2 = parent2[:crossover_point] + parent1[crossover_point:]

                # Apply mutation to both children
                def mutate(child):
                    mutated = []
                    for gene in child:
                        if random.random() < self.mu:
                            mutated.append(random.choice(self.alphabet))
                        else:
                            mutated.append(gene)
                    return mutated

                child1 = mutate(child1)
                child2 = mutate(child2)

                # Add children to new_population
                new_population.append(child1)
                new_population.append(child2)

            # Replace population with new_population
            self.population = new_population

            # Log average self.fitness
            current_fitness = [self.fitness(individual) for individual in self.population]
            avg_fit = sum(current_fitness) / len(current_fitness)
            self.avg_fitness.append(avg_fit)

            # Check if target is reached
            if any("".join(ind) == self.target for ind in self.population):
                print(f"Target reached in {generation + 1} generations!")
                break

            if generation > self.max_generations:
                print("Max generations reached without finding the solution!")
                break

            if generation % self.generations_to_analyse == 0:
                self.analyse_diversity()

        self.analyse_diversity()
        return self.population, generation


if __name__ == "__main__":
    english_letters = [
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
        "g",
        "h",
        "i",
        "j",
        "k",
        "l",
        "m",
        "n",
        "o",
        "p",
        "q",
        "r",
        "s",
        "t",
        "u",
        "v",
        "w",
        "x",
        "y",
        "z",
        "A",
        "B",
        "C",
        "D",
        "E",
        "F",
        "G",
        "H",
        "I",
        "J",
        "K",
        "L",
        "M",
        "N",
        "O",
        "P",
        "Q",
        "R",
        "S",
        "T",
        "U",
        "V",
        "W",
        "X",
        "Y",
        "Z",
    ]
    # res, fits, generation = string_genetic_algorithm(english_letters, "abcdefghijklmno", 15, 200, 0.06, 2)

    ga = GeneticAlgorithm(english_letters, "abcdefghijklmno", 15, 200, 0.06, 2)
    res, generation = ga.run()
    fits = ga.get_avg_fitness()
    hamming_distances = ga.get_hamming_distances()
