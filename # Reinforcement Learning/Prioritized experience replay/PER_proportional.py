import numpy as np
from itertools import tee


"""

tee : 

    - 하나의 iterator를 n개의 iterator로 복사한다. 
     
        tee(iterator, n)
next :

    - iter 함수와 달리 next는 기본값을 설정할 수 있다. 
        iter의 경우 StopIteration trigger인 sentinel이 발견되면 참조가 끊어진다.
        반면 next는 반복이 끝났을 때 기본값을 출력하게 된다.
        
            next(iterator, 기본값)
             
"""

# Build data structure for storing prioritized experiences
# *reference : https://yeonyeon.tistory.com/155
# *reference : https://mrsyee.github.io/rl/2019/01/25/PER-sumtree/
class SumTree: # Binary Heap 대신 SumTree 사용해도 된다. PER 논문에서는 simple binary heap 썼다.
    def __init__(self, capacity):
        self.data_pointer = 0
        self.capacity     = capacity

        self.data         = np.zeros(capacity, dtype = object)
        self.tree         = np.zeros(2 * capacity - 1) # sum tree는 본래 배열의 2배의 메모리 공간이 필요


