import random

def ga(l: int = 5, mu: float = 0.05, max_generations: int = 100):
    # Initialize a random bit string
    current = [random.randint(0, 1) for _ in range(l)]
    generation = 0

    # Fitness function: count the number of ones
    def fitness(bit_string):
        return sum(bit_string)

    # Repeat until the goal is reached
    while True:
        # Create a mutated copy of the current bit string
        mutated = []
        for bit in current:
            if random.random() < mu:
                mutated.append(1 - bit)  # Flip the bit
            else:
                mutated.append(bit)

        # Selection (1+1 GA)
        current_fit = fitness(current)
        mutated_fit = fitness(mutated)

        # Check if the mutated string is closer to the goal (has more ones)
        if mutated_fit > current_fit:
            current = mutated.copy()

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
    print(f"Fitness = {fitness(current)}")

if __name__ == "__main__":
    ga()