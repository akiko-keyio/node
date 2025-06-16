import hashlib
import warnings


def test_node_hash_collision(flow_factory, monkeypatch):
    def dummy_blake2b(data, digest_size=16):
        class Dummy:
            def hexdigest(self):
                return "0" * 32

        return Dummy()

    monkeypatch.setattr(hashlib, "blake2b", dummy_blake2b)

    flow = flow_factory()

    @flow.node()
    def ident(x):
        return x

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        n1 = ident(1)
        n2 = ident(2)

    assert n1 is not n2
    assert n1._hash == n2._hash
    assert any(r.category is RuntimeWarning for r in rec)
