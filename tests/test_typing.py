from typing import assert_type
from flax_typed import jit, value_and_grad


def test_jit():
    def f(x: int) -> str:
        return str(x)

    jitted_f = jit(f)
    assert_type(jitted_f(3), str)


def test_value_and_grad():
    def f(x: float) -> float:
        return x * 2

    grad_f = value_and_grad(f)
    assert_type(grad_f(3), tuple[float, float])  # Assuming has_aux is False by default
    assert_type(grad_f(3)[0], float)  # Gradient
    assert_type(grad_f(3)[1], float)  # Function value
