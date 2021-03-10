from utils import TreeNode

def sameTree(t1, t2):
    if t1 is None and t2 is None:
        return True
    
    if t1 is None or t2 is None:
        return False

    return t1.data == t2.data and sameTree(t1.left, t2.left) and sameTree(t1.right, t2.right)

def insert(root, node):
    """Insert node into binary search tree with root"""
    if root is None:
        root = node
    elif root.data > node.data:
        if root.left is None:
            root.left = node
        else:
            insert(root.left, node)
    else:
        if root.right is None:
            root.right = node
        else:
            insert(root.right, node)

def lowest_common_ancestor(root, node1, node2):
    if root is None:
        return None
    elif node1.data == root.data or node2.data == root.data:
        return root
    elif (
        (node1.data <= root.data) and (node2.data > root.data)
        ) or (
        (node2.data <= root.data) and (node1.data > root.data)   
    ):
        return root
    elif root.data > max(node1.data, node2.data):
        return lowest_common_ancestor(root.left, node1, node2)
    else:
        return lowest_common_ancestor(root.right, node1, node2)

def is_BST(root):
    """If it is empty, no subtrees, or every node in left sub tree is < root and right subtree is > root"""
    if root is None or (root.left is None and root.right is None):
        return True
    if ((root.left is None) or (root.left.data < root.data)) and ((root.right is None) or (root.right.data > root.data)):
        return True and is_BST(root.left) and is_BST(root.right)
    return False