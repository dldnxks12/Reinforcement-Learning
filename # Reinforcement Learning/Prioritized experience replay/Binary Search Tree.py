"""

Insert
Search
Delete
Pre-order traversal
In-order traversal
Post-order traversal

"""

class TreeNode:
    def __init__(self):
        self.data  = None
        self.left  = None
        self.right = None

memory = []
root   = None
Array = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

node = TreeNode()
node.data = Array[0]

root = node
memory.append(node)

# Insertion
for num in Array[1:]:

    node = TreeNode()
    node.data = num

    current = root

    while True:
        if current.data < num:
            if current.left == None:
                current.left = node  # Insert num in here
                break
            current = current.left

        else:
            if current.right == None:
                current.right = node # Insert num in here
                break
            current = current.right

    memory.append(node)

# Search

findNum = 1
current = root

while True:
    if current.data == findNum:
        print("Here")
        break
    elif current.data < findNum:
        if current.data == None:
            print("NONE")
            break
        current = current.left
    else:
        if current.data == None:
            print("NONE")
            break
        current = current.right

# Delete

deleteNum = 8

current = root
parent  = None

while True:
    if current.data == deleteNum:
        if current.left == None and current.right == None:
            if parent.left == current:
                parent.left = None
            else:
                parent.right = None
            print("Delete Done")
            del current
            break
        elif current.left != None and current.right == None:
            if parent.left == current:
                parent.left = current.left
            else:
                parent.right = current.left
            print("Delete Done")
            del current
            break
        elif current.left == None and current.right != None:
            if parent.left == current:
                parent.left = current.right
            else:
                parent.right = current.right
            print("Delete Done")
            del current
            break
    elif current.data < deleteNum:
        if current.left == None:
            print("NONE")
            break
        parent  = current
        current = current.left

    elif current.data > deleteNum:
        if current.right == None:
            print("NONE")
            break
        parent  = current
        current = current.right




