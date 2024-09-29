import re
import heapq
from typing import List, Tuple, Dict, Any

def preprocess_text(text: str) -> List[str]:
    """Preprocess the input text by splitting into sentences and removing punctuation."""
    sentences = re.split(r'(?<=[.!?]) +', text.lower())
    return [re.sub(r'[^\w\s]', '', sentence) for sentence in sentences]

def levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate the Levenshtein distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    if not s2:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

def get_neighbors(state: Tuple[int, int, int], doc1_sentences: List[str], doc2_sentences: List[str]) -> List[Tuple[Tuple[int, int, int], int]]:
    """Get neighboring states and their costs."""
    i, j, g = state
    if i < len(doc1_sentences) and j < len(doc2_sentences):
        cost = levenshtein_distance(doc1_sentences[i], doc2_sentences[j])
        return [((i + 1, j + 1, g + cost), cost)]
    return []

def estimate_heuristic(state: Tuple[int, int, int], doc1_sentences: List[str], doc2_sentences: List[str]) -> int:
    """Estimate the heuristic cost to reach the goal state."""
    i, j, _ = state
    return abs((len(doc1_sentences) - i) - (len(doc2_sentences) - j))

def reconstruct_path(came_from: Dict[Tuple[int, int, int], Tuple[int, int, int]], goal_state: Tuple[int, int, int]) -> List[Tuple[int, int, int]]:
    """Reconstruct the path from start to goal state."""
    path = []
    current_state = goal_state
    while current_state in came_from:
        path.append(current_state)
        current_state = came_from[current_state]
    return path[::-1]

def a_star_search(doc1_sentences: List[str], doc2_sentences: List[str]) -> List[Tuple[int, int, int]]:
    """Perform A* search to find the optimal alignment between two documents."""
    start_state = (0, 0, 0)
    goal_state = (len(doc1_sentences), len(doc2_sentences))
    open_list = [(0, start_state)]
    came_from: Dict[Tuple[int, int, int], Tuple[int, int, int]] = {}
    cost_so_far: Dict[Tuple[int, int, int], int] = {start_state: 0}

    while open_list:
        _, current_state = heapq.heappop(open_list)
        i, j, g = current_state

        if (i, j) == goal_state:
            return reconstruct_path(came_from, current_state)

        for next_state, cost in get_neighbors(current_state, doc1_sentences, doc2_sentences):
            new_cost = g + cost
            if next_state not in cost_so_far or new_cost < cost_so_far[next_state]:
                cost_so_far[next_state] = new_cost
                priority = new_cost + estimate_heuristic(next_state, doc1_sentences, doc2_sentences)
                heapq.heappush(open_list, (priority, next_state))
                came_from[next_state] = current_state

    return []  # Return an empty path if no solution is found

def detect_plagiarism(doc1: str, doc2: str, threshold: int = 5) -> List[Tuple[str, str, int]]:
    """Detect potential plagiarism between two documents."""
    doc1_sentences = preprocess_text(doc1)
    doc2_sentences = preprocess_text(doc2)
    alignment = a_star_search(doc1_sentences, doc2_sentences)
    
    potential_plagiarism = []
    for i, j, _ in alignment:
        if i > 0 and j > 0:
            distance = levenshtein_distance(doc1_sentences[i-1], doc2_sentences[j-1])
            print(f"Comparing: Doc1 - '{doc1_sentences[i-1]}' | Doc2 - '{doc2_sentences[j-1]}' | Distance: {distance}")
            if distance <= threshold:
                potential_plagiarism.append((doc1_sentences[i-1], doc2_sentences[j-1], distance))
    
    print("\nPotential Plagiarism Detected:", potential_plagiarism)
    return potential_plagiarism

def main():
    doc1 = "This is a simple document. It contains a few sentences. Plagiarism is not good."
    doc2 = "This is a simple document. It has a few sentences. Plagiarism is bad."
    
    plagiarized_pairs = detect_plagiarism(doc1, doc2)
    
    print("\nDetected Plagiarism Pairs:")
    for pair in plagiarized_pairs:
        print(f"Doc1: '{pair[0]}'\nDoc2: '{pair[1]}'\nEdit Distance: {pair[2]}\n")

    # Analysis of plagiarism detection results
    if not plagiarized_pairs:
        print("No plagiarism detected. The documents appear to be original.")
    else:
        plagiarism_count = len(plagiarized_pairs)
        total_sentences = len(preprocess_text(doc1))
        plagiarism_percentage = (plagiarism_count / total_sentences) * 100

        print(f"\nPlagiarism Analysis:")
        print(f"Total sentences in Document 1: {total_sentences}")
        print(f"Number of potentially plagiarized sentences: {plagiarism_count}")
        print(f"Percentage of potential plagiarism: {plagiarism_percentage:.2f}%")

        if plagiarism_percentage < 10:
            print("Assessment: Low level of potential plagiarism detected.")
        elif 10 <= plagiarism_percentage < 30:
            print("Assessment: Moderate level of potential plagiarism detected. Further investigation recommended.")
        else:
            print("Assessment: High level of potential plagiarism detected. Thorough review necessary.")

        print("\nRecommendation:")
        if plagiarism_percentage < 10:
            print("The detected similarities are minimal and may be coincidental. No immediate action required.")
        elif 10 <= plagiarism_percentage < 30:
            print("There are notable similarities between the documents. Consider reviewing the flagged sentences and their context.")
        else:
            print("Significant similarities found. A detailed comparison of the documents is strongly advised.")

if _name_ == "_main_":
    main()
