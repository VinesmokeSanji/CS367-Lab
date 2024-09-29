import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from queue import PriorityQueue

class ImageReconstructor:
    def __init__(self, file_path: str, grid_size: int = 4):
        self.grid_size = grid_size
        self.image = self.load_image(file_path)
        self.patches = self.split_image()
        
    def load_image(self, file_path: str) -> np.ndarray:
        with open(file_path, "r") as f:
            data = [int(line.strip()) for line in f.readlines()[1:] if line.strip()]
        return np.array(data).reshape((512, 512)).T
    
    def split_image(self) -> List[np.ndarray]:
        patch_size = self.image.shape[0] // self.grid_size
        return [self.image[i:i+patch_size, j:j+patch_size] 
                for i in range(0, self.image.shape[0], patch_size)
                for j in range(0, self.image.shape[1], patch_size)]
    
    def calculate_difference(self, patch1: np.ndarray, patch2: np.ndarray, direction: str) -> float:
        if direction == 'horizontal':
            return np.sum(np.abs(patch1[:, -1] - patch2[:, 0]))
        elif direction == 'vertical':
            return np.sum(np.abs(patch1[-1, :] - patch2[0, :]))
        else:
            raise ValueError("Direction must be 'horizontal' or 'vertical'")
    
    def find_best_neighbor(self, current: int, used: List[int], direction: str) -> Tuple[float, int]:
        pq = PriorityQueue()
        for i, patch in enumerate(self.patches):
            if i not in used:
                diff = self.calculate_difference(self.patches[current], patch, direction)
                pq.put((diff, i))
        return pq.get()
    
    def reconstruct(self) -> List[List[int]]:
        used = []
        grid = [[0 for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if i == 0 and j == 0:
                    current = np.random.randint(len(self.patches))
                elif j == 0:
                    _, current = self.find_best_neighbor(grid[i-1][j], used, 'vertical')
                else:
                    _, current = self.find_best_neighbor(grid[i][j-1], used, 'horizontal')
                
                grid[i][j] = current
                used.append(current)
        
        return grid
    
    def assemble_image(self, grid: List[List[int]]) -> np.ndarray:
        patch_size = self.image.shape[0] // self.grid_size
        reconstructed = np.zeros_like(self.image)
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                reconstructed[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size] = self.patches[grid[i][j]]
        return reconstructed
    
    def display_image(self, image: np.ndarray):
        plt.imshow(image, cmap="gray")
        plt.title("Reconstructed Image")
        plt.colorbar()
        plt.show()
    
    def run(self):
        print("Original patch arrangement:")
        print([[i for i in range(j, j+self.grid_size)] for j in range(0, self.grid_size**2, self.grid_size)])
        
        reconstructed_grid = self.reconstruct()
        print("\nReconstructed patch arrangement:")
        print(reconstructed_grid)
        
        reconstructed_image = self.assemble_image(reconstructed_grid)
        self.display_image(reconstructed_image)

if __name__ == "__main__":
    reconstructor = ImageReconstructor("scrambled_lena.mat")
    reconstructor.run()
