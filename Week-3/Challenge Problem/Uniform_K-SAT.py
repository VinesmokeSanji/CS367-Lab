import string
import random
import numpy as np

def generate_ksat_instance(num_vars, clause_size, num_clauses):
    """Generate a k-SAT problem instance."""
    positive_vars = list(string.ascii_lowercase)[:num_vars]
    negative_vars = [v.upper() for v in positive_vars]
    all_vars = positive_vars + negative_vars
    
    clauses = []
    max_attempts = 50
    attempts = 0
    
    while len(clauses) < num_clauses and attempts < max_attempts:
        clause = tuple(random.sample(all_vars, clause_size))
        if clause not in clauses:
            clauses.append(list(clause))
            attempts = 0
        else:
            attempts += 1
    
    return all_vars, clauses

def create_random_assignment(variables, num_vars):
    """Create a random truth assignment for variables."""
    truth_values = np.random.randint(2, size=num_vars)
    negated_values = 1 - truth_values
    assignments = np.concatenate((truth_values, negated_values))
    return dict(zip(variables, assignments))

def evaluate_clause(clause, assignment):
    """Evaluate if a clause is satisfied under the given assignment."""
    return any(assignment[var] for var in clause)

def count_satisfied_clauses(problem, assignment):
    """Count the number of satisfied clauses in the problem."""
    return sum(evaluate_clause(clause, assignment) for clause in problem)

# Example usage
n, k, m = 6, 3, 4  # number of variables, clause size, number of clauses

# Generate problem
variables, problem = generate_ksat_instance(n, k, m)
print("Variables:", variables)
print("Problem:")
for clause in problem:
    print(clause)

# Create and evaluate a random assignment
assignment = create_random_assignment(variables, n)
print("\nRandom Assignment:", assignment)

satisfied_count = count_satisfied_clauses(problem, assignment)
print(f"Number of satisfied clauses: {satisfied_count} out of {len(problem)}")

# Simple text-based representation of problem structure
print("\nProblem Structure:")
for i, clause in enumerate(problem):
    print(f"Clause {i}: {' '.join(clause)}")
