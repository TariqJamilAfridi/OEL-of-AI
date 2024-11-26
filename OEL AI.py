import random
import numpy as np

# Parameters
NUM_BEAMS = 3
ANGLE_RANGE = (0, 360)  # Beam angles in degrees
INTENSITY_RANGE = (0, 100)  # Beam intensities
POPULATION_SIZE = 20
GENERATIONS = 100
MUTATION_RATE = 0.1
w1, w2 = 1.0, 1.5  # Weights for cost function

# Simulated healthy tissue and target dose data
random.seed(42)
np.random.seed(42)
healthy_tissue = np.random.uniform(0, 1, size=(NUM_BEAMS,))
target_cells = np.random.uniform(0, 1, size=(NUM_BEAMS,))


# Cost function: Healthy tissue dose + underdose penalty
def calculate_cost(angles, intensities):
    healthy_dose = sum(healthy_tissue[i] * intensities[i] for i in range(NUM_BEAMS))
    target_dose = sum(target_cells[i] * intensities[i] for i in range(NUM_BEAMS))
    underdose_penalty = max(1.0 - target_dose, 0)  # Target should ideally get 1.0 dose


    # Angle penalty: Penalize closely spaced beams
    angle_penalty = 0
    for i in range(NUM_BEAMS):
        for j in range(i + 1, NUM_BEAMS):
            angle_difference = abs(angles[i] - angles[j])
            if angle_difference < 30:  # Penalize if angles are too close (e.g., < 30 degrees)
                angle_penalty += (30 - angle_difference) / 30  # Normalize penalty

    # Total cost
    return w1 * (healthy_dose + angle_penalty) + w2 * underdose_penalty




# Fitness function: Inverse of cost
def fitness_function(chromosome):
    angles = chromosome[:NUM_BEAMS]
    intensities = chromosome[NUM_BEAMS:]
    cost = calculate_cost(angles, intensities)
    return 1 / cost if cost > 0 else 0


# Generate initial population
def generate_population():
    return [[random.uniform(*ANGLE_RANGE) if i < NUM_BEAMS else random.uniform(*INTENSITY_RANGE)
             for i in range(2 * NUM_BEAMS)] for _ in range(POPULATION_SIZE)]


# Selection: Roulette wheel selection
def select_parents(population):
    fitness_values = [fitness_function(chromosome) for chromosome in population]
    total_fitness = sum(fitness_values)
    if total_fitness == 0:
        return random.choices(population, k=2)  # Handle no fitness case
    probabilities = [f / total_fitness for f in fitness_values]
    return random.choices(population, weights=probabilities, k=2)


# Crossover: Uniform crossover
def crossover(parent1, parent2):
    child = [parent1[i] if random.random() < 0.5 else parent2[i] for i in range(len(parent1))]
    return child


# Mutation: Randomly adjust angles or intensities
def mutate(chromosome):
    for i in range(len(chromosome)):
        if random.random() < MUTATION_RATE:
            if i < NUM_BEAMS:  # Mutate angle
                chromosome[i] = random.uniform(*ANGLE_RANGE)
            else:  # Mutate intensity
                chromosome[i] = random.uniform(*INTENSITY_RANGE)
    return chromosome


# Genetic Algorithm
def genetic_algorithm():
    population = generate_population()
    best_chromosome = None
    best_fitness = float('-inf')

    for generation in range(GENERATIONS):
        # Evaluate fitness
        fitness_values = [fitness_function(chromosome) for chromosome in population]
        if max(fitness_values) > best_fitness:
            best_fitness = max(fitness_values)
            best_chromosome = population[fitness_values.index(best_fitness)]

        print(f"Generation {generation}: Best fitness = {best_fitness:.5f}")

        # Create next generation
        new_population = population[:2]  # Elitism: retain top 2
        while len(new_population) < POPULATION_SIZE:
            parent1, parent2 = select_parents(population)
            child = crossover(parent1, parent2)
            new_population.append(mutate(child))

        population = new_population

    return best_chromosome, 1 / best_fitness


# Run the Genetic Algorithm
best_solution, best_cost = genetic_algorithm()
best_angles = best_solution[:NUM_BEAMS]
best_intensities = best_solution[NUM_BEAMS:]
print(f"Optimal beam angles: {best_angles}")
print(f"Optimal beam intensities: {best_intensities}")
print(f"Minimum cost: {best_cost:.2f}")