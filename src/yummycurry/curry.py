import functools
import inspect
import itertools
from typing import Any, Callable, Dict, Iterable, Collection, Optional, Tuple, TypeVar

__all__ = ['curry', 'curry_classmethod', 'curry_method', 'CurriedBase', 'Curried']

A = TypeVar('A')
Args = Tuple
Kwargs = Dict[str, Any]


def _count_positional(params: Iterable[inspect.Parameter], /) -> Tuple[int, bool, bool]:
    positional_kinds = (inspect.Parameter.POSITIONAL_ONLY,
                        inspect.Parameter.POSITIONAL_OR_KEYWORD)
    var_arg_kind = inspect.Parameter.VAR_POSITIONAL
    var_kwarg_kind = inspect.Parameter.VAR_KEYWORD
    count = 0
    var_args = False
    var_kwargs = False
    for param in params:
        if param.kind in positional_kinds:
            count += 1
        elif param.kind == var_arg_kind:
            var_args = True
        elif param.kind == var_kwarg_kind:
            var_kwargs = True
    return count, var_args, var_kwargs


def _partition(left: Collection[A], right: Collection[A], /) -> Tuple[Iterable[A], Iterable[A], Iterable[A]]:
    new_left = (x for x in left if x not in right)
    new_both = (x for x in left if x in right)
    new_right = (x for x in right if x not in left)
    return new_left, new_both, new_right


class CurriedBase:
    __slots__ = ()  # Important for multiple inheritance.


class Curried(CurriedBase):
    __slots__ = 'func', 'args', 'keywords', '__signature__', '__dict__', '__weakref__'
    # These arguments are named like in ``functools.partial`` objects.
    # I am planning for some kind of monad, making it easy to ``join``
    # things like ``Curried(partial(f)) -> Curried(f)`` in the future.
    func: Callable
    args: Args
    keywords: Kwargs
    __signature__: inspect.Signature

    def __new__(cls, f: Callable, old_args: Args, old_kwargs: Kwargs, sig: inspect.Signature, /):
        if not callable(f):
            raise TypeError(f'expected callable, got {type(f)}')
        if isinstance(f, CurriedBase):
            raise TypeError('function should not have been an instance of'
                            'Curried, ``curry()`` should have avoided that')
        self = super().__new__(cls)
        setattr(self, 'func', f)  # Silence MyPy which doesn't like when we assign to a public callable.
        # https://github.com/python/mypy/issues/2427
        self.args = tuple(old_args)  # In case of subclass.
        self.keywords = dict(old_kwargs)  # Copy to avoid mutations.
        self.__signature__ = sig
        functools.update_wrapper(self, f)
        return self

    def __call__(self, *new_args, **new_kwargs):
        return _curry(self.func, self.args + new_args, {**self.keywords, **new_kwargs})

    def __repr__(self) -> str:
        return '{}({}, {}, {}, {})'.format(
            type(self).__name__,
            repr(self.func),
            repr(self.args),
            repr(self.keywords),
            repr(self.__signature__),
        )

    @staticmethod
    def _show(x):
        """Helper for __str__"""
        return str(x) if isinstance(x, Curried) else repr(x)

    def __str__(self) -> str:
        str_f = getattr(self, '__name__')
        if self.args or self.keywords:
            return '{}({})'.format(
                str_f,
                ', '.join(
                    itertools.chain(
                        map(self._show, self.args),
                        ('{}={}'.format(k, self._show(v)) for k, v in self.keywords.items()),
                    )
                )
            )
        return str_f

    def __getnewargs__(self):
        return self.func, self.args, self.keywords, self.__signature__


def _curry(f: Any, args: Args = (), kwargs: Optional[Kwargs] = None, /) -> Any:
    kwargs = kwargs or {}
    if not callable(f):
        # There is nothing more to evaluate.
        if args or kwargs:
            raise TypeError(
                f'{type(f)} is not callable and there are left-over arguments '
                f'at the end of evaluation: *{args}, **{kwargs}'
            )
        return f

    if isinstance(f, Curried):
        # Instead of returning ``Curried(Curried(f))`` we return ``Curried(f)``.
        # That's monadic.
        return f(*args, **kwargs)

    sig = inspect.signature(f)
    params = sig.parameters
    na, var_args, var_kwargs = _count_positional(params.values())
    # Match keyword arguments.
    _, kbn, krn = _partition(params, kwargs, )
    ky: Kwargs = {name: kwargs[name] for name in kbn}
    kv: Kwargs = {}
    kn: Kwargs = {name: kwargs[name] for name in krn}
    if var_kwargs:
        kv, kn = kn, kv
    # Match positional arguments.
    ay: Args = args[:na]
    av: Args = args[:0]
    an: Args = args[na:]
    if var_args:
        av, an = an, av

    try:
        # Does the function have enough parameters to be evaluated?
        sig.bind(*ay, *av, **ky, **kv)
    except TypeError:
        # No it does not (or something is wrong with the signature).
        pass
    else:
        # Yes, we can finally evaluate the function.
        result = f(*ay, *av, **ky, **kv)
        # That result could also be a function, in which case we curry it
        # with our leftover parameters.
        return _curry(result, an, kn)

    # We are not ready to call the function.
    # Just ask for more parameters.
    bound = sig.bind_partial(*ay, *av, **ky, **kv)
    wanted = [p for p in params.values() if p.name not in bound.arguments]
    new_sig = sig.replace(parameters=wanted)
    return Curried(f, args, kwargs, new_sig)


def curry(f: Any, /, *args, **kwargs) -> Any:
    return _curry(f, args, kwargs)


def _check_parameterless(f: Curried, /) -> Curried:
    if f.args or f.keywords:
        raise ValueError(
            f'expected argument-less Curried object, got {f.args}, {f.keywords}'
        )
    return f


class curry_method:
    __slots__ = '_method',
    _method: Curried

    def __new__(cls, f, /):
        self = super().__new__(cls)
        self._method = _check_parameterless(curry(f))
        return self

    def __get__(self, instance, owner=None):
        if instance is None:
            return _check_parameterless(self._method)
        return self._method(instance)

    def __getnewargs__(self):
        _check_parameterless(self._method)
        return self._method.func,

    @property
    def __isabstractmethod__(self) -> bool:
        return getattr(self._method, '__isabstractmethod__')


class curry_classmethod:
    """Place this decorator above @classmethod"""
    __slots__ = '_method',
    _method: classmethod

    def __new__(cls, f, /):
        if not isinstance(f, classmethod):
            raise TypeError(f'expected classmethod, got {type(f)}')
        self = super().__new__(cls)
        self._method = f
        return self

    def __get__(self, instance, owner=None):
        if owner is None:
            owner = type(instance)
        return curry(getattr(self._method, '__func__'), owner)

    def __getnewargs__(self):
        return self._method,

    @property
    def __isabstractmethod__(self) -> bool:
        return getattr(self._method, '__isabstractmethod__')
