==========
YummyCurry
==========

Automatic currying, uncurrying and application of functions and methods
=======================================================================



Features
--------

* Decorators for functions, methods and class methods.
* Supports positional, positional-only, keyword and keyword-only arguments.
* Accepts too few arguments (as do ``functools.partial`` and all other currying
  packages, this is the bare minimum for currying).
* Accepts too many arguments, storing them for the next resulting function that
  wants them.
* Automatically applies the underlying callable when all the necessary arguments
  have been passed (unlike ``functools.partial``).
* Automatically re-curries/re-applies when the result of the callable is
  itself callable (unlike ``functool.partials``).
* Picklable (no lambdas).
* Flat (``curry(curry(f))`` is simplified to ``curry(f)``).
* Inspection-friendly: implements ``__signature__``.



Walkthrough
-----------

Decorator or simple function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The function ``yummycurry.curry`` can be used as a decorator or as a function.

    from yummycurry import curry

    def dbl(x):
        return x*2
    dbl = curry(dbl)  # As a function.

    @curry  # As a decorator.
    def inc(x):
        return x+1


Too few arguments
^^^^^^^^^^^^^^^^^

A trivial use of ``curry`` is to call a function with fewer arguments than it
requires.

We can see it the other way around and design with ``curry`` in mind, in order
to define functions that take more parameters than they actually need.
It is common to see function composition implemented as such::

    def compose(f, g):
        return lambda x: f(g(x))

One severe problem is that lambdas cannot be pickled, which prevents them
from being shared easily in a multiprocessing environment.
Another problem is the lack of ``__doc__`` and ``__name__`` which make
introspection, documentation and pretty-printing difficult.
Finally they are difficult to read.
As a rule of thumb, lambdas should not escape the scope in which they are
defined.

You can avoid returning lambdas by making ``compose`` take a third argument and
relying on ``curry`` to wait for it::

    @curry
    def compose(f, g, x):
        # Composition of unary functions.
        # No need to return a lambda, ``curry`` takes care of it.
        return f(g(x))

    dbl_inc = compose(dbl, inc)
    assert dbl_inc(10) == 22


Automatic application, re-currying and uncurrying
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Automatic application
,,,,,,,,,,,,,,,,,,,,,

With ``functools.partial``, there are two explicit phases:

1. The currying phase:
   create a ``partial`` object by setting some or all the arguments.
2. The application phase:
   apply the partial objects by calling it with all the remaining arguments,
   even if there are actually no remaining argument.

Example::

    from functools import partial

    def cool(x, y, z):
        return x * 100 + y * 10 + z

    p = partial(cool, 1, 2, 3)  # Phase 1: explicit currying.
    result = p()  # Phase 2: explicit application.
    assert result == 123

If we want to curry again we have to be explicit::

    p = partial(cool, 1)  # Explicit currying.
    p = partial(p, 2)  # Explicit currying, again.
    result = p(3)  # Explicit application.
    assert result == 123

With ``yummycurry``, this is automated::

    p = curry(cool, 1)
    p = p(2)
    result = p(3)
    assert result == 123

To achieve this, ``yummycurry`` inspects its underlying function (in our case
``cool``) and compares its signature with the arguments that have been
provided so far.
If the arguments satisfy the signature of the underlying function, then
it is automatically applied, otherwise ``yummycurry`` returns a callable that
waits for more arguments: it re-curries itself.

Automatic re-currying
,,,,,,,,,,,,,,,,,,,,,

Not only does ``yummycurry`` re-curries its underlying callable when it needs
more arguments, but it also automatically curry any callable resulting from
an an application.

If a callable ``f0`` returns a callable ``f1`` that is not explicitly
curried, then ``curry(f0)`` will automatically curry ``f1``::

    def f0(x:int):  # Uncurried
        def f1(y:int, z:int) -> int:  # Uncurried
            return x*100 + y*10 + z
        return f1

    # Without currying this is the only thing that works:
    assert f0(1)(2, 3) == 123

	try:
        assert f0(1)(2)(3) == 123
    except TypeError:
        pass  # The result of f0(1) is not curried.

    # If we curry f0, then its result ``f0(1)`` is automatically curried:
    f0 = curry(f0)
    assert f0(1)(2)(3) == 123  # Now it works.

The process continues: if ``curry(f1)`` returns a callable ``f2`` then it gets
curried as well.
The process stops when the result of a function is not callable.
In this example, the number ``123`` is not callable so the automatic
currying and application stops.

Automatic uncurrying
,,,,,,,,,,,,,,,,,,,,

Unlike ``functools.partial`` and many other Python packages that ship a currying
function, ``yummycurry`` accepts arguments even when they do not match any
parameter of the curried callable.

If a function ``f0`` is called with too many arguments, and if its result is a
function ``f1``, then ``f1`` is automatically called with the arguments that
``f0`` did not use.
From a mathematical point of view, it is not really currying but uncurrying::

    a -> (b -> c)  ===uncurry==>  (a, b) -> c

The process repeats itself automatically until we run out of arguments or the
result is not callable.

    def one_argument_only(x):
        def i_eat_leftovers(y):
            return x + y
        return i_eat_leftovers

    try:
        greeting = one_argument_only('hello ', 'world')
    except TypeError:
        pass  # We knew it would not work.

With ``yummycurry``, that call is valid, the argument ``'world'`` is not used
by ``one_argument`` and is given to its result, which is::

    greet = curry(one_argument_only)
    greeting = greet('hello ', 'world')
    assert greeting == 'hello world'

The three following snippets are equivalent; they all use ``curry``

    greet = curry(one_argument_only)
    greeting = greet('hello ', 'world')

    greet = curry(one_argument_only, 'hello ')
    greeting = greet('world')

    greeting = curry(one_argument_only, 'hello ', 'world')

Automatic function application stops when the result is not callable.
This means that ``curry`` accepts non-callable objects; it just returns
them untouched::

    s = "Don't call us, we'll call you"
    assert curry(s) == s

    @curry
    def actually_constant():
        return 123

    assert actually_constant == 123

It is an error to have left-over arguments when the automatic application stops::

    assert curry(inc, 123) == 124

    curry(inc, 123, 456, x=789)
    # TypeError: left-over arguments at the end of evaluation: *(456,), **{'x':789}
    # Because inc(123) == 124 which is not callable and therefore would not
    # know what to do with ``456``.


Keyword arguments
^^^^^^^^^^^^^^^^^

Use keyword arguments when the order of the positional parameters is
inconvenient (except for positional-only parameters in Python >=3.8)::

    @curry
    def list_map(f, iterable):
        return list(map(f, iterable))

    primes = [2, 3, 5, 7]

    over_primes = list_map(iterable=primes)

    assert over_primes(inc) == [3, 4, 6, 8]

Conflicts between keyword and positional arguments
,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,

Keyword arguments and positional arguments can fight over names.
The ``curry`` function is designed to break whenever Python would break (with
error messages close to the original ones).

* For example, if a positional-only parameter (Python >=3.8) is fed by
  a keyword argument, both ``curry`` and undecorated functions
  raise ``TypeError``.
* If a positional-or-keyword parameter is fed both by a positional and
  a keyword argument, ``TypeError`` is raised.

    @curry
    def give_name(who, name, verbose=False):
        if verbose:
            print('Hello', name)
        new_who = {**who, 'name':name}
        return new_who

    @curry
    def create_genius(iq: int, best_quality:str, *, verbose=False):
        you = dict(iq = 50, awesome_at=best_quality)
        if iq > you['iq']:
            you['iq'] = iq
            if verbose:
                print('Boosting your iq to', iq)
        else:
            if verbose:
                print('You are already smart enough')
        return give_name(you)

Consider the following call::

    dear_reader = create_genius('spitting fire', name='Darling', iq=160, verbose=True)

That call raises ``TypeError: multiple values for argument 'iq'``, as it would
if it were not decorated with ``@curry``.
It would have been possible to make ``curry`` detect
that ``iq`` is passed as a keyword,
and conclude that ``'spitting fire'`` should go to ``best_quality``,
but this would make the decorated and undecorated versions behave differently.
Indeed, Python complains in this situation for the undecorated function.
In order to be transparent and predictable, ``curry`` complains as well.

One could think that doing it in two steps resolves the ambiguity::

    smart = create_genius(name='Darling', iq=160, verbose=True)
    dear_reader = smart('spitting fire')

but it does not.
In this case, the signature of ``smart`` is ``(best_quality: str)``,
and we properly call it with a string.
Nevertheless it still raises the same ``TypeError`` about ``iq`` having more
than one value.
This is by design.
The order of the keyword arguments, and the number of calls that sets them,
should not matter.  If it breaks in one case, it breaks in all cases.  Otherwise
that is a debugging nightmare.

There are many ways to fix this call.
For example, if we insist in passing ``name`` and ``iq`` as keywords, then
it is necessary to pass ``best_quality` as a keyword as well to remove all
ambiguity.
This can be done in any order, in as many calls as wanted::

    dear_reader = create_genius(
        best_quality='spitting fire',
        name='Darling',
        iq=160,
        verbose=True
    )

    # ... equivalent to ...

    smart = create_genius(name='Darling', iq=160, verbose=True)
    dear_reader = smart(best_quality='spitting fire')

Summary: ``curry`` behaves like normal Python would.

Keyword arguments are used only once
,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,

If you run the code above, you will notice that setting ``verbose=True`` makes
``create_genius`` print something.
However, ``give_name`` does not print anything.
This happens because ``curry`` uses arguments only once.
When ``create_genius`` returns the ``give_name`` function, the ``verbose``
argument has already been consumed.







Curried functions are easy on the eyes when given to ``str``.
This is achieved by using the ``__name__`` attribute of underlying callables::

    @curry
    def inc(x: int) -> int:
        return x + 1

    @curry
    def dbl(x: int) -> int:
        return x * 2

    def _compose(f: Callable[[int], int], g: Callable[[int], int], x: int) -> int:
        return f(g(x))

    compose = curry(_compose)  # __name__ will have the underscore.

    assert str(compose(inc, dbl)) == '_compose(inc, dbl)'  # Note the underscore.
    assert str(compose(inc, x=10)) == '_compose(inc, x=10)'

Meanwhile, using ``__repr__`` reveals that the composed function is in fact
an object of type ``Curried``::

    print(repr(compose(inc, x=10))
    # Curried(<function compose at 0x7f7440dd5310>, (Curried(<function inc at
    # 0x7f7440dd51f0>, (), {}, <Signature (x)>),), {'x': 10}, <Signature (g)>)

That ``Curried`` object can be disassembled in the same way
``functools.partial`` objects can, with the attributes ``func``, ``args`` and
``keywords``::

    i10 = compose(inc, x=10)
    assert i10.func == _compose
    assert i10.args == (inc,)
    assert i10.keywords == dict(x=10)

The ``Curried`` object also updates its signature to reflect the parameters
that it still needs.
In our example, the callable ``i10`` (our Curried object), still expects a
parameter ``g`` which is a function from ``int`` to ``int``.
The signature can be accessed via the ``__signature__`` attribute, which is
compatible with ``inspect.signature``::

    import inspect

    assert i10.__signature__ == inspect.signature(i10)
    print(i10.__signature__)  # (g: Callable[[int], int]) -> int

Note that static type checking tools like MyPy_ are unlikely to understand this,
as they look at the code but do not execute it.

.. _MyPy: http://mypy-lang.org/

Under the hood, ``curry`` compares the result of ``inspect.signature`` to the
positional and keyword arguments collected so far.
As soon as the function can be called, it is called.
This means that ``curry`` does not wait when a parameter has a default value::

    @curry
    def increase(x:int, increment:int=1):
        return x + increment

    assert increase(10) == 11  # Does not wait for ``increment``.

    assert increase(10, increment=100) == 110

    inc_100 = increase(increment=100)
    assert inc_100(10) == 110

Instance and class methods can also be curried::

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
            return [impulse, target, self.ears]

    thumper = Rabbit(2, 1)
    monster = Rabbit(3, 2)

    thumperize = Rabbit.breed(thumper)
    oh_god_no = thumperize(monster)  # Currying a class method.
    assert oh_god_no.ears == 2.5
    assert oh_god_no.tails == 2

    thumper_jump = thumper.jump('slow')
    assert thumper_jump('west') == ['slow', 'west', 2]

And of course, you can curry the class itself::

    rabbit = curry(Rabbit)
    deaf = rabbit(0)
    beethoven = deaf(10)  # 5 per hand.
    assert beethoven.ears == 0
    assert beethoven.tails == 10

