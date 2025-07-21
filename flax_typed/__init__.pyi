from collections.abc import Callable, Iterable, Sequence
from typing import Protocol, overload, Any
import jax
from flax.nnx.transforms.autodiff import DiffState, AxisName

class JitWrapped[**P, R]:
    """A function ready to be traced, lowered, and compiled.

    This protocol reflects the output of functions such as
    ``jax.jit``. Calling it results in JIT (just-in-time) lowering,
    compilation, and execution. It can also be explicitly lowered prior
    to compilation, and the result compiled prior to execution.
    """

    def __init__(
        self,
        fun: Callable[P, R],
        in_shardings: Any,
        out_shardings: Any,
        static_argnums: int | Sequence[int] | None = None,
        static_argnames: str | Iterable[str] | None = None,
        donate_argnums: int | Sequence[int] | None = None,
        donate_argnames: str | Iterable[str] | None = None,
        keep_unused: bool = False,
        device: jax.Device | None = None,
        backend: str | None = None,
        inline: bool = False,
        abstracted_axes: Any = None,
    ) -> None: ...

    # implement descriptor protocol so that we can use this as a method
    def __get__(self, obj, objtype=None): ...
    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R: ...
    def eval_shape(self, *args, **kwargs): ...
    def trace(self, *args, **kwargs) -> Traced:
        """Trace this function explicitly for the given arguments.

        A traced function is staged out of Python and translated to a jaxpr. It is
        ready for lowering but not yet lowered.

        Returns:
          A ``Traced`` instance representing the tracing.
        """
        ...

    def lower(self, *args, **kwargs) -> Lowered:
        """Lower this function explicitly for the given arguments.

        This is a shortcut for ``self.trace(*args, **kwargs).lower()``.

        A lowered function is staged out of Python and translated to a
        compiler's input language, possibly in a backend-dependent
        manner. It is ready for compilation but not yet compiled.

        Returns:
          A ``Lowered`` instance representing the lowering.
        """
        ...

class _PreserveFunction(Protocol):
    def __call__[**P, R](self, f: Callable[P, R]) -> Callable[P, R]: ...


@overload
def jit[**P, R](
    *,
    in_shardings: Any = None,
    out_shardings: Any = None,
    static_argnums: int | Sequence[int] | None = None,
    static_argnames: str | Iterable[str] | None = None,
    donate_argnums: int | Sequence[int] | None = None,
    donate_argnames: str | Iterable[str] | None = None,
    keep_unused: bool = False,
    device: jax.Device | None = None,
    backend: str | None = None,
    inline: bool = False,
    abstracted_axes: Any = None,
) -> Callable[[Callable[P, R]], JitWrapped[P, R]]: ...
@overload
def jit[**P, R](
    fun: Callable[P, R],
    *,
    in_shardings: Any = None,
    out_shardings: Any = None,
    static_argnums: int | Sequence[int] | None = None,
    static_argnames: str | Iterable[str] | None = None,
    donate_argnums: int | Sequence[int] | None = None,
    donate_argnames: str | Iterable[str] | None = None,
    keep_unused: bool = False,
    device: jax.Device | None = None,
    backend: str | None = None,
    inline: bool = False,
    abstracted_axes: Any | None = None,
) -> JitWrapped[P, R]: ...

@overload
def value_and_grad[**P, R](
    f: Callable[P, R],
    *,
    argnums: int | DiffState | Sequence[int | DiffState] = 0,
    has_aux: bool = False,
    holomorphic: bool = False,
    allow_int: bool = False,
    reduce_axes: Sequence[AxisName] = (),
) -> Callable[P, tuple[R, R]]: ...
@overload
def value_and_grad[**P, R](
    *,
    argnums: int | DiffState | Sequence[int | DiffState] = 0,
    has_aux: bool = False,
    holomorphic: bool = False,
    allow_int: bool = False,
    reduce_axes: Sequence[AxisName] = (),
) -> Callable[[Callable[P, R]], Callable[P, tuple[R, R]]]: ...
