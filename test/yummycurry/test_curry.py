import inspect
import pickle
import weakref
from functools import partial
from typing import Callable

import pytest

from yummycurry import curry, curry_classmethod, curry_method


def derp(a: int, b: str, c: float) -> float:
    """The most useful function ever"""
    return float(a * len(b) + c)


def hello() -> str:
    return 'hello'


def helloer() -> Callable[[], str]:
    return hello


def succ(n: int) -> int:
    return n + 1


def succer() -> Callable[[int], int]:
    return succ


def one_param(x):
    def other_param(y):
        return x + y

    return other_param


def list_map(f, iterable):
    return list(map(f, iterable))


class Processor:
    """A silly class"""
    __x: int

    def __init__(self, x: int) -> None:
        self.__x = x

    def __repr__(self) -> str:
        return '{}({})'.format(
            self.__class__.__name__,
            repr(self.__x)
        )

    def process(self, y: int, z: int) -> int:
        return self.__x * 100 + y * 10 + z

    @curry_method
    def cprocess(self, y: int, z: int) -> int:
        """Perform a process curryally"""
        return self.__x * 100 + y * 10 + z

    @curry_method
    def zero(self):
        return 0

    @curry_classmethod
    @classmethod
    def from_sum(cls, x, y):
        return Processor(x + y)


@curry
def process(x):
    return x.process


def test_non_callable():
    assert curry(5) == 5


# def test_builtin_function():
#     cstr = curry(str)
#     assert cstr(123) == '123'


def test_callable_with_no_parameter():
    assert curry(hello) == 'hello'
    assert curry(partial(hello)) == 'hello'


def test_result_is_parameterless():
    assert curry(helloer) == 'hello'
    assert curry(partial(helloer)) == 'hello'


def test_already_satisfied():
    assert curry(succ, 10) == 11
    assert curry(succer, 10) == 11


def test_result_takes_one_parameter():
    assert curry(succ)(10) == 11
    assert curry(succer)(10) == 11
    assert curry(partial(succ))(10) == 11
    assert curry(partial(succer))(10) == 11


def test_auto_apply_when_all_parameters_given():
    d = curry(derp)
    d = d(b='two')
    d = d(1)
    d = d(c=3.0)
    assert d == 6.0


def test_keep_currying_when_result_is_callable():
    map_succ = curry(list_map, succ)
    assert map_succ([0, 1, 2]) == [1, 2, 3]
    map_succ = curry(list_map)(succ)
    assert map_succ([0, 1, 2]) == [1, 2, 3]


def test_extra_params_get_passed():
    f = curry(one_param)
    assert f(10, 100) == 110


def test_complain_leftover_params():
    with pytest.raises(TypeError) as e:
        curry('hello', 1)
    assert 'left-over arguments at the end of evaluation' in str(e)


def test_kwargs_absorbs():
    @curry
    def black_hole(mass, **slurp):
        return 'singularity'

    assert black_hole(10, curvature='thicc') == 'singularity'


def test_args_absorbs():
    @curry
    def thief(name, *your_things):
        return 'haha'

    assert thief('Butch Cassidy', 'wallet', 'dreams') == 'haha'


def test_nested_calls():
    @curry
    def add(x: int, /, y: int = 1) -> int:
        return x + y

    primes = [2, 3, 5, 7]
    expected = [7, 8, 10, 12]
    cur_list_map = curry(list_map)

    add5 = add(y=5)
    over_primes = cur_list_map(iterable=primes)
    assert over_primes(add5) == expected

    map_plus5 = cur_list_map(add5)
    assert map_plus5(primes) == expected

    f = cur_list_map(add(y=5))
    assert f(primes) == expected


def test_signature():
    assert inspect.signature(curry(succ)) == inspect.signature(succ)


def test_monadic_join():
    csucc = curry(succ)
    ccsucc = curry(csucc)
    assert not isinstance(ccsucc.func, type(ccsucc))
    assert ccsucc.func == csucc.func == succ


def test_str():
    cderp = curry(derp)
    assert str(cderp) == 'derp'
    assert str(cderp(1, b='2')) == "derp(1, b='2')"

    clambda = curry(lambda a, b: a * b)
    assert str(clambda) == '<lambda>'
    assert str(clambda(10)) == '<lambda>(10)'

    assert str(curry(list_map, curry(succ))) == 'list_map(succ)'


def test_pickle():
    src = curry(derp)
    data = pickle.dumps(src)
    new = pickle.loads(data)
    assert str(src) == str(new)
    #
    src = Processor(1)
    data = pickle.dumps(src)
    new = pickle.loads(data)
    assert str(src) == str(new)


def test_weakref():
    src = curry(derp)
    ref = weakref.ref(src)
    assert ref() == src


def test_uncurried_method():
    o1 = Processor(1)
    o9 = Processor(9)
    process1 = curry(o1.process)
    process9 = curry(o9.process)
    assert process1(2, 3) == 123
    assert process9(2, 3) == 923
    assert process(o1, 2, 3) == 123
    assert process(o9, 2, 3) == 923


def test_curried_method():
    o1 = Processor(1)
    assert o1.cprocess(2)(3) == 123
    assert Processor.cprocess(o1)(2)(3) == 123
    assert o1.zero == 0


def test_curried_classmethod():
    o1 = Processor.from_sum(3)
    o2 = o1(6)
    assert o2.cprocess(2)(3) == 923


def test_dunders():
    cderp = curry(derp)
    for name in '__module__ __name__ __qualname__ __doc__ __annotations__'.split():
        assert getattr(cderp, name) == getattr(derp, name)


def test_readme_examples():
    def dbl(x):
        return x * 2

    dbl = curry(dbl)  # As a function.

    @curry  # As a decorator.
    def inc(x):
        return x + 1

    # ------------

    @curry
    def compose(f, g, x):
        """Composition of unary functions."""
        # No need to return a lambda, ``curry`` takes care of it.
        return f(g(x))

    dbl_inc = compose(dbl, inc)
    assert dbl_inc(10) == 22

    # Function composition is associative: as long as the order or the leaves
    # is preserved, the way that the tree forks does not matter.
    pipeline_1 = compose(compose(dbl, compose(inc, dbl)), compose(inc, inc))
    pipeline_2 = compose(compose(compose(compose(dbl, inc), dbl), inc), inc)
    assert pipeline_1(10) == 2 * (1 + 2 * (1 + 1 + 10))
    assert pipeline_2(10) == 2 * (1 + 2 * (1 + 1 + 10))

    # ------------

    from functools import partial

    def cool(x, y, z):
        return x * 100 + y * 10 + z

    p = partial(cool, 1, 2, 3)  # Phase 1: explicit currying.
    result = p()  # Phase 2: explicit application, even if there are no arguments.
    assert result == 123

    p = partial(cool, 1)  # Explicit currying.
    p = partial(p, 2)  # Explicit currying, again.
    result = p(3)  # Explicit application.
    assert result == 123

    # ------------

    p = curry(cool, 1)
    p = p(2)
    result = p(3)
    assert result == 123

    s = "Don't call us, we'll call you"
    assert curry(s) == s

    @curry
    def actually_constant():
        return 123

    assert actually_constant == 123

    # ------------

    def f0(x: int):  # Uncurried
        def f1(y: int, z: int) -> int:  # Uncurried
            return x * 100 + y * 10 + z

        return f1

    # Without currying, this is the only thing that works:
    assert f0(1)(2, 3) == 123

    try:
        assert f0(1)(2)(3) == 123
    except TypeError:
        pass  # The result of f0(1) is not curried so f0(1)(2) is incorrect.

    # If we curry f0, then its result ``f0(1)`` is automatically curried:
    f0 = curry(f0)
    assert f0(1)(2)(3) == 123  # Now it works.

    # ------------

    def one_param_only(x):
        def i_eat_leftovers(y):
            return x + y

        return i_eat_leftovers

    try:
        greeting = one_param_only('hello ', 'world')
    except TypeError:
        pass  # We knew it would not work.

    greet = curry(one_param_only)
    greeting = greet('hello ', 'world')
    assert greeting == 'hello world'

    greet = curry(one_param_only)
    greeting = greet('hello ', 'world')
    assert greeting == 'hello world'

    greet = curry(one_param_only, 'hello ')
    greeting = greet('world')
    assert greeting == 'hello world'

    greeting = curry(one_param_only, 'hello ', 'world')
    assert greeting == 'hello world'
    # ------------

    # Good:
    assert curry(inc, 123) == 124

    # Bad:
    try:
        curry(inc, 123, 456, x=789)
    except TypeError:
        pass

    # ------------

    @curry
    def list_map(f, iterable):
        return list(map(f, iterable))

    primes = [2, 3, 5, 7]

    over_primes = list_map(iterable=primes)

    assert over_primes(inc) == [3, 4, 6, 8]

    # ------------

    @curry
    def give_name(who, name, verbose=False):
        if verbose:
            print('Hello', name)
        new_who = {**who, 'name': name}
        return new_who

    @curry
    def create_genius(iq: int, best_quality: str, *, verbose=False):
        you = dict(iq=50, awesome_at=best_quality)
        if iq > you['iq']:
            you['iq'] = iq
            if verbose:
                print('Boosting your iq to', iq)
        else:
            if verbose:
                print('You are already smart enough')
        return give_name(you)

    with pytest.raises(TypeError):
        dear_reader = create_genius('spitting fire', name='Darling', iq=160, verbose=True)

    with pytest.raises(TypeError):
        smart = create_genius(name='Darling', iq=160, verbose=True)
        dear_reader = smart('spitting fire')

    dear_reader = create_genius(
        best_quality='spitting fire',
        name='Darling',
        iq=160,
        verbose=True
    )
    smart = create_genius(name='Darling', iq=160, verbose=True)
    dear_reader = smart(best_quality='spitting fire')

    # ------------

    @curry
    def greedy(x, *args):
        if args:
            print('I am stealing your', args)

        def starving(y):
            return x + y

        return starving

    assert greedy(10)(1) == 11

    with pytest.raises(AssertionError):
        assert greedy(10, 1) == 11

    assert greedy(10, 1000, 2000, 3000, 4000)(1) == 11

    @curry
    def black_hole(mass, **slurp):
        def hawking_radiation(*, bleep):
            return 'tiny {}'.format(bleep)

        return hawking_radiation

    assert black_hole(10, bleep='proton', curvature='thicc')(bleep='neutrino') == 'tiny neutrino'

    # ------------

    @curry
    def inc(x: int) -> int:
        return x + 1

    @curry
    def dbl(x: int) -> int:
        return x * 2

    def _compose(f: Callable[[int], int], g: Callable[[int], int], x: int) -> int:
        return f(g(x))

    compose = curry(_compose)  # __name__ will retain the underscore.

    assert str(compose(inc, dbl)) == '_compose(inc, dbl)'  # Note the underscore.
    assert str(compose(inc, x=10)) == '_compose(inc, x=10)'

    # ------------

    i10 = compose(inc, x=10)
    assert i10.func == _compose
    assert i10.args == (inc,)
    assert i10.keywords == dict(x=10)

    assert i10.__signature__ == inspect.signature(i10)

    # ------------

    @curry
    def increase(x: int, increment: int = 1):
        return x + increment

    assert increase(10) == 11  # Does not wait for ``increment``.

    assert increase(10, increment=100) == 110

    inc_100 = increase(increment=100)
    assert inc_100(10) == 110

    # ------------

    class Rabbit:
        def __init__(self, ears, tails):
            self._ears = ears
            self._tails = tails

        @curry_method  # Works here like a read-only property
        def ears(self):
            return self._ears

        @curry_method
        def tails(self):
            return self._tails

        @curry_classmethod
        @classmethod
        def breed(cls, rabbit1, rabbit2):
            # Accurate model of rabbit genetics.
            return cls(
                (rabbit1.ears + rabbit2.ears) / 2,  # Yes, floats.
                rabbit1.tails * rabbit2.tails,
            )

        @curry_method
        def jump(self, impulse, target):
            # Does not mean anything, just a demonstration.
            return [impulse, target, 'boing']

    thumper = Rabbit(2, 1)
    monster = Rabbit(3, 2)

    thumperize = Rabbit.breed(thumper)
    oh_god_no = thumperize(monster)  # Currying a class method.
    assert oh_god_no.ears == 2.5
    assert oh_god_no.tails == 2

    thumper_jump = thumper.jump('slow')
    assert thumper_jump('west') == ['slow', 'west', 'boing']

    rabbit = curry(Rabbit)
    deaf = rabbit(ears=0)
    beethoven = deaf(tails=10)  # 5 per hand.
    assert beethoven.ears == 0
    assert beethoven.tails == 10
