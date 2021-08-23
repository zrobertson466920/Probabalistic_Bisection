"""
This file contains widely used classes and functions.
"""

from typing import List
import math
import numpy as np
from sklearn.datasets import make_spd_matrix

MIN_GRAD = 1e-1

# radius of small sphere when running LPME inside QME
SMALL_SPHERE_RADIUS = 5e-3


class Oracle:
    """
    An oracle parameterized by (hidden) a and B, with optional c

    Scoring function:
    score = <a,r> + 0.5 r^T B r  + c

    In the linear case, set B to 0.
    """

    def __init__(self, a: np.array, B: np.matrix, c: float = 0):
        assert len(a) == B.shape[0] == B.shape[1]
        self.a = a
        self.B = B
        self.c = c

    """
    Applies (hidden) scoring function
    """

    def _score(self, rate_vec: np.array) -> float:
        if len(rate_vec) != len(self.a):
            raise ValueError("mismatched rate vec length")

        rate_matrix = np.matrix(rate_vec)
        return (
            (self.a * rate_vec).sum()
            + (0.5 * rate_matrix @ self.B @ rate_matrix.T)[0, 0]
            + self.c
        )

    """
    Compares rate vectors r1 and r2 and returns which is greater using a hidden scoring function
    
    Returns True if r1 is preferred over r2, otherwise False
    """

    def compare(self, r1: np.array, r2: np.array, alpha:float) -> bool:
        r1_score = self._score(r1)
        r2_score = self._score(r2)
        
        if np.random.rand() >= alpha:
            return r1_score >= r2_score
        else:
            return not (r1_score >= r2_score)
        

class Sphere:
    """
    A sphere in n-dimensions with origin, radius
    """

    def __init__(self, origin: np.array, radius: float, n: float):
        assert len(origin) == n
        self.origin = origin
        self.radius = radius
        self.n = n


class ThetaSearchSpace:
    """
    Defines a search space for an angle
    """

    def __init__(self, start: float, stop: float):
        self.start = start
        self.stop = stop


def normalize(x, to=1.0):
    """
    Used to normalize a vector or matrix such that:
    || x ||_2 = `to`

    Returns normalized x
    """
    return x / np.sqrt(np.sum(np.square(x)) / (to ** 2))


def check_is_bad_grad(v: np.array) -> bool:
    """
    Check if gradient vector `v` is ill-formed.
    """
    return abs(v[0]) < MIN_GRAD


def check_a_B_sphere_satisfy_conditions(
    sphere: Sphere, a: np.array, B: np.matrix
) -> bool:
    """
    REGULARITY ASSUMPTION:
    - make sure all (normalized) gradients that will be estimated by QME algorithm are > MIN_GRAD.
    - make sure the (normalized) gradient at <sphere.radius - SMALL_SPHERE_RADIUS, 0, 0, ... > is
    well-formed with respect to the (normalized) gradient at < -(sphere.radius - SMALL_SPHERE_RADIUS), 0, 0, ... >.
    See why 2) for a specific explanation.

    WHY:
    1) Suppose the true gradient is <0.001, 0.3, 0.4, ...>. LPME could estimate
    <0.0001, 0.3, 0.4, ...>. THis is very close in vector/spherical space,
    but with fractions we see that 0.3/0.001 = 300 << 0.3 / 0.0001 = 3000 (off by 10x factor).
    This would ruin the calculation. The regularity assumption lower bounds each element
    in the (normalized) true gradient so that this kind of error is minimized. In the paper,
    it is treated as a condition that is either satisfied or not. In real life,
    this condition is satisfied "to some degree." We discuss the distribution of fractional errors
    resulting from the choice of MIN_GRAD in the appendix.

    2) The gradient in the first basis direction and the gradient in the negative first basis direction are
    subtracted in qme.py/QMESC to compute the "cmn" variable. See the code and paper for where this calculation
    comes from. Specifically, the denominator looks like this: (f_1_neg0 / f_0_neg0) - (f_1_0 / f_0_0).
    We want this denominator to be non-zero, meaning f_1_neg0 / f_0_neg0 != f_1_0 / f_0_0. We will go
    even further and make sure abs( f_1_neg0 / f_0_neg0 - f_1_0 / f_0_0 ) > MIN_GRAD. This should
    make it more robust to LPME estimation error.
    """
    q = len(a)
    sr = sphere.radius
    d = a + np.array(B @ sphere.origin)[0]
    f_z = normalize(d)
    if check_is_bad_grad(f_z):
        return False

    z_neg0 = np.zeros(q)
    z_neg0[0] = -(sr - SMALL_SPHERE_RADIUS)
    f_neg0 = normalize(d + np.array(B @ z_neg0)[0])
    if check_is_bad_grad(f_neg0):
        return False

    z_0 = np.zeros(q)
    z_0[0] = sr - SMALL_SPHERE_RADIUS
    f_0_opt = normalize(d + np.array(B @ z_0)[0])
    if check_is_bad_grad(f_0_opt):
        return False
    # make sure z_0 and z_neg0 are well-formed wrt to each other
    elif np.abs(f_neg0[1] / f_neg0[0] - f_0_opt[1] / f_0_opt[0]) < MIN_GRAD:
        return False

    # other basis directions
    for i in range(1, q):
        z_i = np.zeros(q)
        z_i[i] = sr - SMALL_SPHERE_RADIUS
        f_i_opt = normalize(d + np.array(B @ z_i)[0])
        if check_is_bad_grad(f_i_opt):
            return False

    return True


def qme_norm(a, B):
    """
    Normalizes a and B so that:
    || a ||_2^2 + || B ||_F^2 = 1.0

    See create_a_B for an explanation of a and B.

    Returns the normalized (a, B)
    """
    # Keeps || a ||_2^2 between 0.3 and 0.7 so that comparisons
    # in elicitation error can be properly done for different
    # choices of q. Without this, a has a tendency to naturally
    # shrink relative to B because B has more terms, meaning
    # higher q's will have smaller a's. Comparing elicitation
    # errors to smaller q's would not be fair.
    amount_a = np.random.uniform(low=0.3, high=0.7)
    amount_b = 1.0 - amount_a
    a_sum = np.square(a).sum()
    B_sum = np.square(B).sum()
    return a * np.sqrt(amount_a / a_sum), B * np.sqrt(amount_b / B_sum)


def create_a_B(sphere: Sphere, q: int, well_formed=True):
    """
    Create a random (a, B, sphere) such that:
    - a is a rank q vector
    - B is a (q x q) positive definite matrix
    - || a ||_2^2 + || B ||_F^2 = 1.0 = 1.0
    - if well_formed=True, normalized true oracle gradient at every QME query point has value > MIN_GRAD
    - otherwise, a and B are randomly generated

    Returns (a, B)
    """
    while True:
        a = np.random.rand(q)
        B = np.matrix(make_spd_matrix(q))
        a, B = qme_norm(a, B)

        if not well_formed:
            # do not check a, B; just return
            return a, B

        # check a, B satisfy well-formed condition; otherwise, loop
        if check_a_B_sphere_satisfy_conditions(sphere, a, B):
            return a, B
