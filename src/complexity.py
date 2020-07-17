from branch import Branch
from binarytree import Node

class Complexity:
    def __init__(self):
        self.root = None
        self.subdivision_trees = []
        self.idct_count = 0
    
    def __str__(self):
        total = sum(tree.size for tree in self.subdivision_trees)
        return "Number of IDCTs: %s\nNumber of interval evaluations: %s" % (self.idct_count, total)

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
    
    def draw(self):
        for tree in self.subdivision_trees:
            print(tree)