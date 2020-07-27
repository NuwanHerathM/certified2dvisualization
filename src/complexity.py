from branch import Branch
from binarytree import Node

import logging

import math

# Logger

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler('complexity.log', 'w')
handler_format = logging.Formatter('%(message)s')
logger.addHandler(handler)

class Complexity:
    def __init__(self, d, n):
        self.root = None
        self.subdivision_trees = []
        self.idct_count = 0
        self.horner_count = 0
        self.clenshaw_count = 0
        self.d = d
        self.n = n
    
    def __str__(self):
        total = sum(tree.size for tree in self.subdivision_trees)
        n_logn =  round(self.n + math.log2(self.n))
        return ("evaluation\t{}\n".format(self.horner_count * self.d + self.clenshaw_count * self.d + self.idct_count * n_logn) +
                "subdivision\t{}".format(total * self.d))
    
    def log(self):
        logger.info(self)

    def endSubdivision(self):
        if self.root is not None:
            self.subdivision_trees.append(self.root)
            self.root = None
    
    def posIntEval(self, branch, node=None):
        if branch == Branch.ROOT:
            self.root = Node(1)
            return self.root
        elif branch == Branch.LEFT:
            node.left = Node(1)
            return node.left
        else:
            node.right = Node(1)
            return node.right
    
    def negIntEval(self, branch, node=None):
        if branch == Branch.ROOT:
            self.root = Node(0)
            return self.root
        elif branch == Branch.LEFT:
            node.left = Node(0)
            return node.left
        else:
            node.right = Node(0)
            return node.right
    
    def incrIDCT(self):
        self.idct_count += 1
    
    def incrHorner(self):
        self.horner_count += 1
    
    def incrClenshaw(self):
        self.clenshaw_count += 1
    
    def draw(self):
        for tree in self.subdivision_trees:
            print(tree)