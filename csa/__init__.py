"""Certified Sparse Attention: runtime-verified fidelity for sparse-attention LLM serving.

Mechanism 1 (label-free divergence detection)   -> csa.signals, csa.paired
Mechanism 2 (sampled dense verification, CS)    -> csa.verify
Mechanism 3 (verification as elastic work)      -> csa.scheduler
Study A measurement harness                     -> csa.paired, csa.tasks, csa.recording
"""

__version__ = "0.1.0"
