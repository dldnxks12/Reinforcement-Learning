
class Calculator:
    def __init__(self):
        self.result = 0

    def add(self, num):
        self.result += num
        return self.result

calc1 = Calculator()
calc2 = Calculator()

print(calc1.add(4))
print(calc1.add(7))
