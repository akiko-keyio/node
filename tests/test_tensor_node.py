import pytest

import node
from node.tensor import Dim, TensorNode


@node.define()
def double(x: int) -> int:
    return x * 2


def test_tensor_node_executes_nodes():
    dim = Dim("item", values=(1, 2, 3))
    nodes = [double(value) for value in dim.values]

    tensor = TensorNode.from_nodes(nodes, dim=dim)

    assert tensor() == [2, 4, 6]


def test_tensor_node_rejects_empty_nodes():
    with pytest.raises(ValueError, match="empty"):
        TensorNode.from_nodes([])


def test_tensor_node_rejects_dim_mismatch():
    dim = Dim("item", values=(1, 2))
    nodes = [double(1), double(2), double(3)]

    with pytest.raises(ValueError, match="length"):
        TensorNode.from_nodes(nodes, dim=dim)


def test_dim_len_returns_values_length():
    dim = Dim("sample", values=("a", "b"))

    assert len(dim) == 2


def test_tensor_node_apply_broadcasts_scalar():
    dim = Dim("item", values=(10, 20, 30))
    tensor = TensorNode.from_nodes([double(value) for value in dim.values], dim=dim)

    @node.define()
    def add(x: int, y: int) -> int:
        return x + y

    result = tensor.apply(add, 5)

    assert result() == [25, 45, 65]


def test_tensor_node_apply_rejects_dim_mismatch():
    dim_a = Dim("item", values=(1, 2))
    dim_b = Dim("item", values=(1, 2, 3))
    a = TensorNode.from_nodes([double(1), double(2)], dim=dim_a)
    b = TensorNode.from_nodes([double(1), double(2), double(3)], dim=dim_b)

    @node.define()
    def add(x: int, y: int) -> int:
        return x + y

    with pytest.raises(ValueError, match="length"):
        a.apply(add, b)


def test_tensor_node_apply_rejects_dim_values_mismatch():
    dim_a = Dim("item", values=(1, 2))
    dim_b = Dim("item", values=(2, 3))
    a = TensorNode.from_nodes([double(1), double(2)], dim=dim_a)
    b = TensorNode.from_nodes([double(2), double(3)], dim=dim_b)

    @node.define()
    def add(x: int, y: int) -> int:
        return x + y

    with pytest.raises(ValueError, match="values"):
        a.apply(add, b)


def test_tensor_node_align_to_reorders_nodes():
    dim_src = Dim("item", values=("b", "a"))
    dim_dst = Dim("item", values=("a", "b"))
    tensor = TensorNode.from_nodes([double(2), double(1)], dim=dim_src)

    aligned = tensor.align_to(dim_dst)

    assert aligned() == [2, 4]


def test_tensor_node_apply_aligns_dim_value_order():
    dim_a = Dim("item", values=("b", "a"))
    dim_b = Dim("item", values=("a", "b"))
    a = TensorNode.from_nodes([double(2), double(1)], dim=dim_a)
    b = TensorNode.from_nodes([double(1), double(2)], dim=dim_b)

    @node.define()
    def add(x: int, y: int) -> int:
        return x + y

    result = a.apply(add, b)

    assert result() == [8, 4]
