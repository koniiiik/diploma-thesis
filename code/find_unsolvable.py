#!/usr/bin/env python

from __future__ import division

import math
import subprocess
import random
import sys

from liblll import (create_matrix_from_knapsack, lll_reduction,
    best_vect_knapsack)


# Mean value of random variable multiplied to the max element when
# generating a random instance.
GEOMETRIC_MEAN = 1.2

# List of solvers defined in here.
SOLVERS = ['liblll', 'ntl']

NTL_SOLVER_BINARY = './knapsack_solver_ntl'


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
    stdout, _ = solver.communicate(input='%s %s\n%s\n' % (
        len(a), m,
        ' '.join(str(x) for x in a),
    ))

    if solver.returncode > 0:
        raise SolutionNotFound()

    result = [int(x) for x in stdout.split()]
    return result[:-1], result[-1]


def solve_knapsack_liblll(a, m):
    """
    a is the list of elements
    m is the expected subset sum
    """
    for curr_m in (m, sum(a) - m):
        matrix = create_matrix_from_knapsack(a, curr_m)
        reduced = lll_reduction(matrix)
        try:
            return (find_solution_vector(reduced, curr_m), int(m == curr_m))
        except SolutionNotFound:
            pass
    raise SolutionNotFound()


def find_solution_vector(matrix, m):
    for v in transposed(matrix):
        v = v[:-1]
        s = sum(v)
        if s == 0:
            continue
        # TODO: this appears to be broken
        expected = s // len([x for x in v if x != 0])
        if all(x == 0 or x == expected for x in v) and s // expected == m:
            return [x // expected for x in v]
    raise SolutionNotFound()


def transposed(m):
    return [[m[j][i] for j in range(len(m))] for i in range(len(m[0]))]


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
    beta = [(r * (1 << i)) % q for i in range(n)]
    return (beta, r)


def create_for_density(n, d):
    """
    Creates a random set of n elements with a density of at most d as
    defined in [LO83].
    """
    max_elem = int(2 ** (n / d) * (1 + random.expovariate(1 / (GEOMETRIC_MEAN - 1))))
    result = random.sample(range(1, max_elem), n - 1)
    result.append(max_elem)
    return result


def sample_from_mask(mask, population):
    """
    Returns a subset and indicator vector of population based on bits from
    mask.
    """
    subset = []
    indicators = [0] * len(population)
    for i, elem in enumerate(population):
        if mask & 1 << i:
            subset.append(elem)
            indicators[i] = 1
    return subset, indicators


if __name__ == "__main__":
    assert len(sys.argv) == 4, "Usage: %s <density> <number of elements> <number of iterations>" % (sys.argv[0],)

    d = float(sys.argv[1])
    n = int(sys.argv[2])
    its = int(sys.argv[3])
    elems = create_for_density(n, d)
    print("Density: %.5f" % (n / math.log(max(elems), 2)))
    print("Elements: %s" % (" ".join("%s" % e for e in elems)))
    options = set(range(1, (1 << n) - 1))
    for i in range(its):
        if not options:
            break
        mask = random.sample(options, 1).pop()
        options.remove(mask)
        subset, indicators = sample_from_mask(mask, elems)
        try:
            results = solve_knapsack(elems, sum(subset))
            print(results)
            continue
            # TODO: Dead code follows, needs reworking
            if result != indicators:
                found_subset = [e for i, e in zip(result, elems) if i]
                if sum(found_subset) != sum(subset):
                    message = "Invalid solution found."
                else:
                    message = "Different solution found."
                print("%s Original: %s; found: %s" % (
                    message,
                    " ".join(sorted("%s" % e for e in subset)),
                    " ".join(sorted("%s" % e for e in found_subset)),
                ))
        except SolutionNotFound:
            print("Unsolved: %s" % (" ".join("%s" % e for e in subset)))
