import Accumulator as thk_accumulator

a = [1, 2, 3]
b = [4, 5, 6]
c = zip(a, b)
print(list(c))

c = zip(a, b)
print([x + y for x, y in c])

print([0.0]*2)

metric = thk_accumulator.Accumulator(2)
metric.add(8,10)
metric.add(9,10)
print(metric.data)
