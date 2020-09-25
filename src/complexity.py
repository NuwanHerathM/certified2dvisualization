from branch import Branch
from binarytree import Node

import logging

import math
import statistics

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
        self.coef_idct = 2
        self.dsc = []
    
    def __str__(self):
        total = sum(tree.size for tree in self.subdivision_trees)
        n_logn =  round(self.n * math.log2(self.n))
        return ("evaluation\t{}\n".format(self.horner_count * self.d + self.clenshaw_count * self.d + self.idct_count * n_logn * self.coef_idct) +
                "subdivision\t{}".format(total * self.d))
    
    def log(self):
        logger.info(self)

    def resetSubdivision(self):
        self.root = None

    def endSubdivision(self):
        if self.root is not None:
            self.subdivision_trees.append(self.root)
    
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
    
    def subTreeSize(self):
        res = 0
        if (self.root != None):
            res = self.root.size
        return res
    
    def _copy_tree(self, tree=None):
        if tree is None:
            tree = self.root

        res = None
        
        def copy(node):
            left = None
            right = None
            if node.left is not None:
                left = copy(node.left)
            if node.right is not None:
                right = copy(node.right)
            return Node(node.value, left, right)
        
        if tree is not None:
            res = copy(tree)
        return res

    def prune(self, tree=None):
        if tree is None:
            tree = self.root

        copy = self._copy_tree(tree)

        def prune_aux(node):
            if node.left is None and node.right is None:
                return node
            if node.left.value == 1:
                node.left = prune_aux(node.left)
            if node.right.value == 1:
                node.right = prune_aux(node.right)
            if node.left.value == 0 and node.right.value == 0:
                node = Node(0)
            return node

        if tree is not None:
            res = prune_aux(copy)
        else:
            res = None
        return res

    def leaves(self):
        # print(self.root)
        print(self.prune().size)
        # print(self.root.leaves)
        # print(self.root.levels[-1])
        print(sum(node.value == 1 for node in self.root.leaves))
    
    def descartes(self, n):
        self.dsc.append(n)
    
    def subdivision_analysis(self):
        with open('complexity_sub.log', 'w') as f:
            for d, s in zip(self.dsc, self.subdivision_trees):
                f.write(f"{d}\t{s.size}\t{self.prune(s).size}\n")
        
        n = len(self.dsc) - 1
        a = statistics.mean(self.dsc)
        b = statistics.mean(map(len, self.subdivision_trees))
        c = statistics.mean(map(lambda x: len(self.prune(x)), self.subdivision_trees))
        with open('complexity_sub_avg.log', 'a') as f:
            f.write(f"{n}\t{a}\t{b}\t{c}\n") 
