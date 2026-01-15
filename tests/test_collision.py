import hashlib

import node


def test_node_hash_equality(runtime_factory, monkeypatch):
    """Test that nodes with same hash are considered equal (no collision detection)."""
    def dummy_blake2b(data, digest_size=16):
        class Dummy:
            def hexdigest(self):
                return "0" * 32

        return Dummy()

    monkeypatch.setattr(hashlib, "blake2b", dummy_blake2b)

    runtime_factory()

    @node.define()
    def ident(x):
        return x

    n1 = ident(1)
    n2 = ident(2)

    # With same forced hash, nodes are considered equal
    assert n1._hash == n2._hash
    assert n1 == n2  # Now equal since we don't do collision detection
