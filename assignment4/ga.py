import random
from typing import List, Callable, Tuple

# Fitness function: count the number of ones
def fitness(bit_string: List[int]) -> int:
    return sum(bit_string)

def mutation_if(current: List[int], mutated: List[int]) -> Tuple[List[int], int]:
    # Selection (1+1 GA)
    current_fit = fitness(current)
    mutated_fit = fitness(mutated)

    # Check if the mutated string is closer to the goal (has more ones)
    if mutated_fit > current_fit:
        current = mutated.copy()
        current_fit = mutated_fit

    return current, current_fit

def mutation_always(current: List[int], mutated: List[int]) -> List[int]:
    return mutated.copy()


def ga(mutation_func: Callable[[List[int], List[int]], List[int]], l: int = 5, mu: float = 0.05, max_generations: int = 100) -> Tuple[List[int], List[int]]:
    # Initialize a random bit string
    current = [random.randint(0, 1) for _ in range(l)]
    generation = 0

    # Repeat until the goal is reached
    fits = []
    while True:
        # Create a mutated copy of the current bit string
        mutated = []
        for bit in current:
            if random.random() < mu:
                mutated.append(1 - bit)  # Flip the bit
            else:
                mutated.append(bit)

        current, current_fit = mutation_func(current, mutated)
        fits.append(current_fit)

        # Check if the goal is reached (all ones)
        if all(bit == 1 for bit in current):
            break

        generation += 1
        if generation > max_generations:
            print("Max generations reached without finding the solution!")
            break

        # Print progress (optional)
        if generation % 10 == 0:
            print(f"Generation {generation}: Current bit string = {current}, Fitness = {current_fit}")

    # Print the final result
    print("\nFinal result:")
    print(f"Bit string = {current}")
    print(f"Fitness = {current_fit}")
    print(f"Generation = {generation}")

    return current, fits

if __name__ == "__main__":
    l = 100
    mu = 1/l
    ga(mutation_if, l, mu, 1500)