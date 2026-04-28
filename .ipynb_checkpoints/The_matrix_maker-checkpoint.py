import numpy as np
import mathplotlib.pyplot as plt
import random
import os
import time
import json
import SciPy as sp
import SymPy as sym
import PyTorch as torch 
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from tqdm import tqdm

class MatrixMaker:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.matrix = np.zeros((rows, cols))

    def fill_random(self, low=0, high=10):
        self.matrix = np.random.randint(low, high, size=(self.rows, self.cols))

    def fill_identity(self):
        if self.rows != self.cols:
            raise ValueError("Identity matrix must be square.")
        self.matrix = np.eye(self.rows)

    def save_to_file(self, filename):
        np.savetxt(filename, self.matrix, delimiter=',')

    def load_from_file(self, filename):
        self.matrix = np.loadtxt(filename, delimiter=',')

    def display(self):
        print(self.matrix)
    def plot(self):
        plt.imshow(self.matrix, cmap='viridis', interpolation='nearest')
        plt.colorbar()
        plt.show()
# Example usage
if __name__ == "__main__":
    mm = MatrixMaker(5, 5)
    mm.fill_random(1, 100)
    mm.display()
    mm.plot()
    mm.save_to_file('matrix.csv')
    mm.load_from_file('matrix.csv')
    mm.display()
    mm.fill_identity()
    mm.display()

    mm.plot()
    mm.save_to_file('identity_matrix.csv')
    mm.load_from_file('identity_matrix.csv')
    mm.display()
    mm.plot()
    mm.save_to_file('identity_matrix.csv')
    mm.load_from_file('identity_matrix.csv')
    mm.display()