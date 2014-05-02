#!/usr/bin/env python

from __future__ import division, print_function

from collections import namedtuple
import itertools
import math
import random
import subprocess
import sys

import click
from liblll import (create_matrix_from_knapsack, lll_reduction,
    best_vect_knapsack)


# Mean value of random variable multiplied to the max element when
# generating a random instance.
GEOMETRIC_MEAN = 1.2

# The maximum value of n (the number of elements) for which to guarantee
# uniqueness of generated instances.
MAX_N_FOR_UNIQUENESS = 20

# List of solvers defined in here.
SOLVERS = ['liblll', 'ntl']
SOLVERS = ['ntl']

NTL_SOLVER_BINARY = './knapsack_solver_ntl'

try:
    xrange(47)
except NameError:
    xrange = range


class cached_property(object):
    """
    Decorator that converts a method with a single self argument into a
    property cached on the instance. Borrowed from Django.
    """
    def __init__(self, func):
        self.func = func

    def __get__(self, instance, type=None):
        if instance is None:
            return self
        res = instance.__dict__[self.func.__name__] = self.func(instance)
        return res


"""
Definitions of solvers.
"""

class SolutionNotFound(Exception):
    pass


def solve_knapsack(a, m):
    solutions = {solver_name: None for solver_name in SOLVERS}
    for solver_name in SOLVERS:
        solver = getattr(sys.modules[__name__], 'solve_knapsack_%s' % solver_name)
        try:
            solutions[solver_name] = solver(a, m)
        except SolutionNotFound:
            pass
    return solutions


def solve_knapsack_ntl(a, m):
    solver = subprocess.Popen(NTL_SOLVER_BINARY, stdin=subprocess.PIPE,
                              stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE)
    stdout, stderr = solver.communicate(input='%s %s\n%s\n' % (
        len(a), m,
        ' '.join(str(x) for x in a),
    ))

    if solver.returncode > 0:
        raise SolutionNotFound()

    # This is a very ugly maybe-workaround for an even uglier heisenbug
    # which appears to happen with a probability of about 1/500000 and is
    # likely caused by some stdout buffering or some such.
    if not stdout:
        stdout = stderr.splitlines()[-1]

    result = [int(x) for x in stdout.split()]
    try:
        return result[:-1], result[-1]
    except Exception, e:
        e.args += (stdout,)
        raise


def solve_knapsack_liblll(a, m):
    """
    a is the list of elements
    m is the expected subset sum
    """
    for curr_m in (m, sum(a) - m):
        matrix = create_matrix_from_knapsack(a, curr_m)
        reduced = lll_reduction(matrix)
        try:
            return (find_solution_vector(reduced, a, curr_m), int(m == curr_m))
        except SolutionNotFound:
            pass
    raise SolutionNotFound()


def find_solution_vector(matrix, a, m):
    for v in transposed(matrix):
        v = v[:-1]
        s = sum(v)
        # All nonzero elements should be 0 < v[i] <= n.
        if not 0 < s <= len(v)**2:
            continue
        expected = s // len([x for x in v if x != 0])
        if (all(x == 0 or x == expected for x in v) and
                sum(a[i] for i, x in enumerate(v) if x > 0) == m):
            return [x // expected for x in v]
    raise SolutionNotFound()


def transposed(m):
    return [[m[j][i] for j in xrange(len(m))] for i in xrange(len(m[0]))]


"""
Definitions of generators.
"""

Instance = namedtuple('Instance', 'mask, elems, sum')

class RandomGeneratorStrategy(object):
    """
    Generator of sequences of elements and random subsets of those
    sequences.
    """

    def __init__(self, n, density, **kwargs):
        self.n = n
        self.density = density
        self.kwargs = kwargs

    def get_max_element(self):
        """
        Returns the maximum element such that the resulting density is at
        most self.density.
        """
        return int(2 ** (self.n / self.density) * (1 + random.expovariate(1 / (GEOMETRIC_MEAN - 1))))

    def get_actual_density(self):
        """
        Returns the density of the generated sequence.
        """
        return self.n / math.log(max(self.elements), 2)

    @cached_property
    def elements(self):
        """
        Creates a random set of n elements with a density of at most d as
        defined in [LO83].
        """
        max_elem = self.get_max_element()
        result = random.sample(xrange(1, max_elem), self.n - 1)
        result.append(max_elem)
        return result

    def mask_to_instance(self, mask=None, indicators=None):
        """
        Converts a mask (either a bit mask in an integer or a sequence of
        indicator variables) to an Instance.
        """
        assert mask is not None or indicators is not None

        if indicators is None:
            indicators = [0] * self.n
            for i in xrange(self.n):
                if mask & 1 << i:
                    indicators[i] = 1

        subseq = indicators_to_elems(indicators, self.elements)
        return Instance(indicators, subseq, sum(subseq))

    def instances(self):
        """
        Generator for individual instances, i.e. subsets of the sequence
        of elements.
        """
        if self.n <= MAX_N_FOR_UNIQUENESS:
            options = set(xrange(1, (1 << self.n) - 1))
            while options:
                mask = random.sample(options, 1).pop()
                options.remove(mask)
                yield self.mask_to_instance(mask=mask)
        else:
            while True:
                mask = random.choice(xrange(1, (1 << self.n) - 1))
                yield self.mask_to_instance(mask=mask)


class LinearlyDependentStrategy(RandomGeneratorStrategy):
    """
    This generator ensures there is always at least a specified number d
    of small linear dependencies in each instance.

    It does so by generating the first n - d elements of the large set and
    then generating the remaining d elements in the following way: for
    each already chosen element x_i, a random lambda_i is chosen with
    exponential distribution and random sign; the new element is then
    chosen as | \sum_{i=1}^{n-d+k} lambda_i x_i |.
    """

    LAMBDA_MEAN = 1.25

    def __init__(self, *args, **kwargs):
        self.dependencies = kwargs.pop('linear_deps')
        super(LinearlyDependentStrategy, self).__init__(*args, **kwargs)

    @cached_property
    def elements(self):
        max_elem = self.get_max_element()
        result = set(random.sample(xrange(1, max_elem),
                                   self.n - 1 - self.dependencies))
        result.add(max_elem)
        lambd = 1/self.LAMBDA_MEAN
        choice, exp = random.choice, random.expovariate
        signs = [-1, 1]
        while len(result) < self.n:
            new_elem = sum(choice(signs) * int(exp(lambd)) * x
                           for x in result)
            if new_elem and new_elem not in result:
                result.add(abs(new_elem))
        result = list(result)
        random.shuffle(result)
        return result


def create_mh_keys(n):
    """
    Creates a MH instance of block size n. The superincreasing sequence is
    always chosen as the first n powers of 2; q is chosen as 2^(n+1) and r
    is a random odd integer lower than q.

    Returns the pair (beta, r) where beta is the public key, i.e. a set of
    elements for unambiguous knapsack instances.
    """
    q = 1 << (n)
    r = random.randrange(1, q, 2)
    beta = [(r * (1 << i)) % q for i in xrange(n)]
    return (beta, r)


STRATEGIES = {
    'random': RandomGeneratorStrategy,
    'linear-dep': LinearlyDependentStrategy,
}


"""
The actual runner.
"""

def indicators_to_elems(indicators, elems):
    return [x for i, x in zip(indicators, elems) if i]


@click.command()
@click.option('--strategy', '-s', type=click.Choice(list(STRATEGIES.keys())),
              required=True,
              help="Strategy for random instance generation.")
@click.option('--density', '-d', type=click.FLOAT, required=True,
              help="Maximum density for generated isntances.")
@click.option('--elements', '-n', type=click.INT, required=True,
              help="The number of elements of A.")
@click.option('--instances', '-i', type=click.INT, required=True,
              help="The number of instances to consider for each set.")
@click.option('--linear-deps', '-l', type=click.INT,
              help="The number of small linear dependencies. Only "
              "applicable for the linear-dep strategy.")
def run(strategy, density, instances, elements, **kwargs):
    generator = STRATEGIES[strategy](elements, density, **kwargs)
    print("Density: %.5f" % generator.get_actual_density())
    print("Elements: %s" % (" ".join("%s" % e for e in generator.elements)))
    for i, instance in itertools.izip(xrange(instances),
                                      generator.instances()):
        try:
            vector, first_round = solve_knapsack_ntl(generator.elements, instance.sum)
            if not first_round:
                vector = [1 - x for x in vector]
            print("Round %d, sum %d: %s" % (
                1 - first_round,
                instance.sum,
                " ".join("%d" % x for x in indicators_to_elems(vector, generator.elements)),
            ))
        except SolutionNotFound:
            print("Failed sum %d: %s" % (
                instance.sum,
                " ".join("%d" % x for x in instance.elems),
            ))
        except Exception, e:
            message = "Uncaught error %r on sum %d: %s" % (
                e,
                instance.sum,
                " ".join("%d" % x for x in instance.elems),
            )
            print(message)
            print(message, file=sys.stderr)


if __name__ == "__main__":
    run()
