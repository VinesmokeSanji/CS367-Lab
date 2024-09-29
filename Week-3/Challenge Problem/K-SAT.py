import random
from typing import List, Dict, Tuple

class KSatSolver:
    def init(self, num_variables: int, clause_length: int, num_clauses: int):
        self.num_variables = num_variables
        self.clause_length = clause_length
        self.num_clauses = num_clauses
        self.problem = self.generate_problem()
        self.current_solution = self.random_solution()

    def generate_problem(self) -> List[List[Tuple[int, bool]]]:
        problem = []
        for _ in range(self.num_clauses):
            clause = []
            for _ in range(self.clause_length):
                var = random.randint(1, self.num_variables)
                polarity = random.choice([True, False])
                clause.append((var, polarity))
            problem.append(clause)
        return problem

    def random_solution(self) -> Dict[int, bool]:
        return {i: random.choice([True, False]) for i in range(1, self.num_variables + 1)}

    def evaluate_solution(self, solution: Dict[int, bool]) -> int:
        satisfied_clauses = 0
        for clause in self.problem:
            if any(solution[var] == polarity for var, polarity in clause):
                satisfied_clauses += 1
        return satisfied_clauses

    def flip_variables(self, solution: Dict[int, bool], num_flips: int) -> Dict[int, bool]:
        new_solution = solution.copy()
        variables_to_flip = random.sample(list(new_solution.keys()), num_flips)
        for var in variables_to_flip:
            new_solution[var] = not new_solution[var]
        return new_solution

    def variable_neighborhood_search(self, max_iterations: int = 1000) -> Dict[int, bool]:
        best_solution = self.current_solution
        best_score = self.evaluate_solution(best_solution)

        for _ in range(max_iterations):
            for neighborhood_size in range(1, self.num_variables // 2):
                candidate = self.flip_variables(best_solution, neighborhood_size)
                candidate_score = self.evaluate_solution(candidate)

                if candidate_score > best_score:
                    best_solution = candidate
                    best_score = candidate_score

                if best_score == self.num_clauses:
                    return best_solution

        return best_solution

    def solve(self) -> Tuple[Dict[int, bool], int]:
        solution = self.variable_neighborhood_search()
        score = self.evaluate_solution(solution)
        return solution, score

# Example usage
if name == "main":
    solver = KSatSolver(num_variables=5, clause_length=3, num_clauses=10)
    solution, score = solver.solve()
    print(f"Best solution: {solution}")
    print(f"Satisfied clauses: {score} out of {solver.num_clauses}")
