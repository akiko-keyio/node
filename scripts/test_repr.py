import node
node.configure(cache=node.MemoryLRU())

@node.dimension(name='time')
def time_dim():
    return [1, 2, 3]

@node.define()
def process(x):
    return x * 10

# Create VectorNode
t = time_dim()
v = process(t)

print("=== Dimension Node repr ===")
print(repr(t))
print()
print("=== VectorNode repr ===")
print(repr(v))
print()
print("=== VectorNode dims ===")
print(f"dims: {v.dims}")
print(f"coords: {v.coords}")
