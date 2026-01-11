import node
from node import sweep, Config

# 配置依赖链：train -> preprocess -> mode
node.configure(config=Config({
    "mode": "fast",
    
    "preprocess": {
        "_target_": "__main__.preprocess", # 指定函数路径
        "method": "${mode}"
    },
    
    "train": {
        "data": "${preprocess}",       # 自动实例化并注入 preprocess 节点
        "strategy": "${mode}"
    }
}))

@node.define()
def preprocess(method):
    return f"Data({method})"

@node.define()
def train(data, strategy):
    return f"Result({data}, {strategy})"

if __name__ == "__main__":
    # 扫描 mode，自动触发 preprocess 和 train 的重建
    results = node.sweep(
        train,
        config={"mode": ["fast", "accurate"]}
    ).get()

    print(f"Results: {results}")

    expected = ["Result(Data(fast), fast)", "Result(Data(accurate), accurate)"]
    assert results == expected, f"Expected {expected}, but got {results}"
    print("Verification successful!")
