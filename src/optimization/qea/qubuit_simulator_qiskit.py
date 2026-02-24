import math
import random
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.visualization import plot_bloch_multivector


class QuBitSim():
  def __init__(self):
    q = QuantumRegister(1)
    c = ClassicalRegister(1)
    self.qc = QuantumCircuit(q,c)
    self.psi = Statevector(self.qc)
  
  def initialize(self):
    self.qc.h(0)
    self.update_statevector()

  def rotate(self, direction = 1, angle=0.01*math.pi):
    if direction * angle != 0:
      self.qc.ry(direction * angle, 0)
      self.update_statevector()

  def draw(self):
    return self.qc.draw(output='mpl')

  def plot_bloch_multivector(self):
    return plot_bloch_multivector(self.qc)

  def update_statevector(self):
    self.psi = Statevector(self.qc)
  
  def draw_statevector(self):
    return self.psi.draw('latex')
  
  def get_probabilities(self):
    return self.psi.probabilities_dict()
  
  def measure(self):
    probabilities = self.get_probabilities()
    random_number = random.random()
    return 0 if random_number < probabilities['0'] else 1
