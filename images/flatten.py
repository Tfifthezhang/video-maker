import functools
import itertools
import operator

import numpy as np
import perfplot


def func_for(a):
    return [item for sublist in a for item in sublist]


def func_sum_brackets(a):
    return sum(a, [])


def func_reduce(a):
    return functools.reduce(operator.concat, a)


def functools_reduce_iconcat(a):
    return functools.reduce(operator.iconcat, a, [])


def itertools_chain(a):
    return list(itertools.chain.from_iterable(a))


def numpy_flat(a):
    return list(np.array(a).flat)


def numpy_concatenate(a):
    return list(np.concatenate(a))


def func_extend(a):
    out = []
    for sublist in a:
        out.extend(sublist)
    return out


b = perfplot.bench(
    setup=lambda n: [list(range(10))] * n,
    # setup=lambda n: [list(range(n))] * 10,
    kernels=[
        func_for,
        func_sum_brackets,
        func_reduce,
        func_extend,
    ],
    n_range=[2 ** k for k in range(16)],
    xlabel="number lists"
)
b.save("out.svg")
b.show()