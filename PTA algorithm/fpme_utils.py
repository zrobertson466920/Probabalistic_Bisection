"""
This file contains classes and functions for FPME.
"""

from typing import List, Tuple
import numpy as np
from sklearn.datasets import make_spd_matrix

from common import (
    Sphere,
    Oracle,
    normalize,
    check_is_bad_grad,
    check_a_B_sphere_satisfy_conditions,
)


class FairOracle:
    """
    A fair oracle with scoring metric s:
    s = (1 - lamb) * performance + 1/2 * lamb * fairness
    performance = <a, \sum_{g} T_g * r_g>
    fairness = \sum_{g1, g2} ( r_{g1} - r_{g2} )^T B_{g1,g2} ( r_{g1} - r_{g2} )

    Where:
    r_g: a rate vector representing P(h = i | G = g, Y = j). This is originally a rate matrix and is flattened into a
    vector of the off-diagonal elements.

    a: a q-dimensional linear performance metric; it can be though of as a cost associated with P(h = i | Y = j). This
    is originally a confusion matrix and is flattened into a vector of the off-diagonal elements.

    T_g := a vector of length q that describes P(G = g | Y = i). Again think about a flattened confusion matrix.
        Note that the vector is flattened off-diag rates. In the original rate matrix, row i would mean P(G = g | Y = i)
        and is constant for the row. So a single T_g will look like:
        < P(G = g | Y = 1), P(G = g | Y = 1), ... (nc - 1 times), P(G = g | Y = 2), P(G = g | Y = 2), ... >

    lamb: float between 0 and 1; specifies trade-off between fairness term and performance term. Specifically, lamb
        is how much you care about fairness with lamb=0 meaning fairness does not matter and lamb=1 meaning performance
        does not matter.

    B_{g1, g2}: qxq matrix of fairness violation costs. Must be symmetric positive-definite.
    """

    def __init__(
        self, a: np.array, B: List[List[np.matrix]], lamb: float, T: List[np.array]
    ):
        """
        Most of these are defined in the class docstring.
        a :=  a q-dimensional linear performance metric
        B := a list of all B_{g1, g2}. See class docstring for more details.
        lamb: float between 0 and 1; specifies trade-off between fairness term and performance term
        T := a matrix of `T_g` vectors. See class docstring for more details.
        """
        assert len(a) == B[0][0].shape[0] == B[0][0].shape[1]
        assert (np.sum(T, axis=0) - 1 < 1e-3).all()
        assert 0 <= lamb <= 1
        assert len(T) == len(B) == len(B[0])

        self.a = a
        self.B = B
        self.lamb = lamb
        self.T = T

        # number of groups
        self.num_g = len(T)

    def _score(self, rate_list: List[np.array]) -> float:
        """
        Applies hidden scoring metric. Smaller the better.
        """
        if len(rate_list) != self.num_g:
            raise ValueError("mismatched rate list number of groups")
        if len(rate_list[0]) != len(self.a):
            raise ValueError("mismatched rate vec length")

        r = np.zeros(len(self.a))
        for r_g, t_g in zip(rate_list, self.T):
            r += r_g * t_g
        p = np.dot(self.a, r)  # performance metric value

        f = 0.0  # fairness metric value
        for i in range(self.num_g):
            for j in range(i + 1, self.num_g):
                r_i = rate_list[i]
                r_j = rate_list[j]
                B_ij = self.B[i][j]
                discrep = r_i - r_j
                f += (discrep.T @ B_ij @ discrep)[0, 0]

        return (1 - self.lamb) * p + 0.5 * self.lamb * f

    def compare(self, r1_list: List[np.array], r2_list: List[np.array], alpha:float=0) -> bool:
        """
        Performs a comparison between r1_list and r2_list. Returns True if
        r1_list is preferred over r2_list, otherwise False.
        """
        r1_score = self._score(r1_list)
        r2_score = self._score(r2_list)

        if np.random.rand() >= alpha:
            return r1_score >= r2_score
        else:
            return not (r1_score >= r2_score)
        # return r1_score >= r2_score


class Fair_QME_Parameterization:
    """
    Holds two groups, i and j, as "included" in the parameterization. Example:
    3 groups: 0, 1, 2

    If the "included" groups are 0 and 2, then FPME, will do __
    """

    def __init__(self, i: int, j: int):
        self.i = i
        self.j = j

    def __repr__(self) -> str:
        return f"({self.i}, {self.j})"


class FairParametizer:
    """
    In FPME, we have a Bij matrix for groups i and j. This means there are `ng` C 2
    matrices total, if there are `ng` groups. We need to create `ng` C 2 distinct
    parameterizations. The algorithm will assign unique rates to everything "included"
    in the parameterization, and the trivial rate to everything not included.
    See Fair_QME_Parameterization for an example.

    nc := number of classes
    q := length of off-diag rate vectors, should be nc**2 - nc
    ng := number of groups
    """

    def __init__(self, q: int, ng: int):
        self.q = q
        self.ng = ng
        self.v = ng * (ng - 1) // 2  # ng C 2

    """
    ng := number of groups
    v := ng C 2
    
    Creates a list of v parameterizations. Typically, this means setting two 
    classes to rate s and the others to some trivial rate. This method
    will simply enumerate all ways to do that.
    
    However, for two choices of ng this will not work: ng = 2 and ng = 4
    
    In the ng = 2 case, we pick one class 1 be s and the other to be the trivial.
    
    In the ng = 4 case, consider <s, s, o, o>. Note that the B vectors associated
    with this are B02, B03, B12, B13 (all s,o pairs). Now consider <o, o, s, s>. The
    B vectors are the exact same! In the ng = 4 case, enumerating (4 C 2) creates 
    duplicate uses of the B vectors. This is not a problem for any other number of groups.
    """

    def create_paramaterization(self) -> List[Fair_QME_Parameterization]:
        if self.ng == 2:
            # -1 means ignore
            return [Fair_QME_Parameterization(0, -1)]

        elif self.ng == 4:
            return [
                # -1 means ignore
                Fair_QME_Parameterization(0, -1),
                Fair_QME_Parameterization(1, -1),
                Fair_QME_Parameterization(2, -1),
                Fair_QME_Parameterization(3, -1),
                # uses both variables
                Fair_QME_Parameterization(0, 3),
                Fair_QME_Parameterization(1, 3),
            ]

        else:
            params = []
            for i in range(self.ng):
                for j in range(i + 1, self.ng):
                    p = Fair_QME_Parameterization(i, j)
                    params.append(p)
            return params

    def recover_B_lambda_from_measurements(
        self, b_measurements: List[np.matrix], lparams: List[Fair_QME_Parameterization]
    ) -> Tuple[List[List[np.matrix]], float]:
        """
        Determines the matrix that created `b_measurements` using `lparams`. Solves
        the system of equations (matrices) and recovers B. Because \lambda
        is the normalization factor, \lambda can be solved for too.
        """
        assert len(b_measurements) == len(lparams)
        mm = self.get_B_matrix_for_parameterizations(lparams)
        mm_inv = mm ** -1

        # all B_ij's are in terms of (1 - lamb) / lamb * B_ij
        # solves system of equations
        lamb_b_hat = [
            [
                np.matrix(np.zeros((self.q, self.q), dtype=np.float64))
                for _ in range(self.ng)
            ]
            for _ in range(self.ng)
        ]
        for i in range(self.v):
            li, lj = self.index_to_coord(i)
            for j in range(self.v):
                lamb_b_hat[li][lj] += mm_inv[i, j] * b_measurements[j]
            lamb_b_hat[lj][li] = lamb_b_hat[li][lj]

        b_norms = 0
        for i in range(self.ng):
            for j in range(i + 1, self.ng):
                b_norms += np.linalg.norm(lamb_b_hat[i][j])

        # (1. - lamb) / lamb * b_norms = 2
        # lamb = 1 / (1 + 1 / b_norms)
        lamb_hat = 1 / (1.0 + 2.0 / b_norms)
        for i in range(self.ng):
            for j in range(i + 1, self.ng):
                lamb_b_hat[i][j] *= (1.0 - lamb_hat) / lamb_hat
                lamb_b_hat[j][i] = lamb_b_hat[i][j]
        B_hat = lamb_b_hat
        return B_hat, lamb_hat

    def get_B_matrix_for_parameterizations(
        self, lparams: List[Fair_QME_Parameterization]
    ):
        """
        v := (ng C 2)
        lparams := a list of v paramaterizations

        Returns a (v, v) invertible matrix M where row i corresponds to
        a vector of 0's and 1's indicating which B vectors were used by the parameterization.

        Example:
        param = Fair_QME_Parameterization(0, 1)
        ng = 3
        vector: <0, 1, 1>

        The first entry represents whether B_01 was used. Second entry is B_02. Third is B_12.
        Since groups 0 and 1 are included, the B matrices chosen will be B_02 and B_12.
        Hence, those are set to 1 in the vector.
        """
        assert len(lparams) == self.v
        mm = np.matrix(np.zeros((self.v, self.v)))
        for r in range(self.v):
            p = lparams[r]
            for k in range(self.ng):
                if k != p.i and k != p.j:
                    if p.i == -1:
                        mm[r, self.coord_to_index(k, p.j)] = 1.0
                    elif p.j == -1:
                        mm[r, self.coord_to_index(k, p.i)] = 1.0
                    else:
                        mm[r, self.coord_to_index(k, p.i)] = 1.0
                        mm[r, self.coord_to_index(k, p.j)] = 1.0

        return mm

    def coord_to_index(self, cx, cy):
        """
        PRIOR TO READING CODE:
        Just know that this function and index_to_coord help us when working with
        the invertible matrix M in the paper. We need a way to map into M[i][j] to/from the
        traditional B, which is a 2D list of matrices with 0s on the diagonal and B[i][j] = B[j][i].

        In a vector of length (ng C 2), this finds the index for some (cx, cy) given cy > cx
        Otherwise, it swaps them because the vector is supposed to be a concise
        representation of a symmetric matrix where V[i][j] = V[j][i]

        Derivation:
        Given desired coordinate (x,y)
        index = (\sum_{i=0}^{x-1} ng - 1 - i)+ (y - x) - 1

        Write out some examples to see why. Note that in the type of matrix we are representing,
        the first interesting coordinate is (0,1), then (0,2), ...
        """
        assert cy < self.ng, print(f"given ng: {self.ng}, cx: {cx}, cy: {cy}")
        if cy < cx:
            return self.coord_to_index(cy, cx)
        return (2 * self.ng - 1 - cx) * cx // 2 + (cy - cx) - 1

    def index_to_coord(self, i: int):
        """
        See docstring for coord_to_index for more info.

        In a vector of length ng C 2, this finds the (cx, cy) given an index.
        The vector is supposed to be a concise representation of a symmetric matrix
        where V[i][j] = V[j][i]

        Recovering the index from the equation shown in coord_to_index involves
        finding the max x, then figuring out y.
        """
        # first find row
        x = int(
            ((2 * self.ng - 1) / 2) - np.sqrt(-2 * i + ((1 - 2 * self.ng) / 2) ** 2)
        )
        # find where are positionally at start of row
        idx = self.coord_to_index(x, x + 1)
        # i - idx is how much more to go, and x + 1 is the first valid y coordinate in the row
        return (x, i - idx + x + 1)


class Reparam_LPME_FairOracle:
    """
    Reparameterize a fair oracle as a linear oracle. Defined as:
    <s> -> <s, s, ...> (for the total number of groups)
    """

    def __init__(self, fair_oracle: FairOracle, ng: int):
        """
        fair_oracle: a FairOracle
        ng: number of groups
        """
        self.fair_oracle = fair_oracle
        self.ng = ng

    def compare(self, r1, r2, alpha) -> bool:
        """
        LPME will call this method, but before it is passed to the fair_oracle
        it is reparameterized so that the fair_oracle can properly process it.
        """
        r1 = self.reparam_lpme(r1)
        r2 = self.reparam_lpme(r2)
        return self.fair_oracle.compare(r1, r2, alpha)

    def reparam_lpme(self, vec: np.array):
        """
        Reparamterizes a vector (which LPME is querying) into fair_oracle space.
        """
        return [vec for u in range(self.ng)]

    def convert_to_LPME_Oracle(self) -> Oracle:
        """
        This class is just a reparameterization of the fair oracle. This method
        returns the linear Oracle it is equivalent too.
        """
        a_lpme = (1.0 - self.fair_oracle.lamb) * self.fair_oracle.a
        return Oracle(a_lpme, np.matrix(np.zeros((len(a_lpme), len(a_lpme)))))


class Reparam_QME_FairOracle:
    """
    Reparamterize a fair oracle as a QME oracle. Defined as:
    <s> -> <o, o, ..., s, o, o, ..., s, o, o, ...>

    Where each s is mapped to two groups. The other groups get the trivial rate tr.
    The choice of the groups is determined by fqp, a Fair_QME_Parameterization.
    """

    def __init__(
        self,
        fair_oracle: FairOracle,
        nc: int,
        q: int,
        ng: int,
        fqp: Fair_QME_Parameterization,
        tr: np.array,
    ):
        """
        fair_oracle: a FairOracle
        nc: number of classes
        q: number of off-diag elements in rate vector, should be nc**2 - nc
        ng: number of groups
        fqp: a Fair_QME_Parameterization that defines the paramterization
        tr: trivial rate
        """
        self.fair_oracle = fair_oracle
        self.nc = nc
        self.q = q
        self.ng = ng
        self.fqp = fqp
        self.tr = tr

    def compare(self, r1, r2, alpha) -> bool:
        """
        This is called by the QME algorithm thinking this is a QME oracle.
        It is passed to the fair oracle after applying paramterization.
        """
        r1 = self.reparam_qme(r1)
        r2 = self.reparam_qme(r2)
        return self.fair_oracle.compare(r1, r2, alpha)

    def _score(self, r1) -> float:
        """
        This should never be called by QME, LPME, FPME. It is simply for debugging
        """
        r1 = self.reparam_qme(r1)
        return self.fair_oracle._score(r1)

    def reparam_qme(self, vec: np.array):
        """
        Reparameterizes a QME query into fair_oracle space. Described in __init__.
        """
        reparam = [self.tr for _ in range(self.ng)]
        if self.fqp.i != -1:
            reparam[self.fqp.i] = vec + self.tr
        if self.fqp.j != -1:
            reparam[self.fqp.j] = vec + self.tr

        return reparam

    def convert_to_QME_Oracle(self) -> Oracle:
        """
        This entire class is just a reparameterization of the fair oracle as a quadratic oracle.
        This method returns the (quadratic) Oracle object that it is equivalent too.
        """
        t_sum = np.zeros(self.q)
        if self.fqp.i != -1:
            t_sum += self.fair_oracle.T[self.fqp.i]
        if self.fqp.j != -1:
            t_sum += self.fair_oracle.T[self.fqp.j]

        b_sum = np.matrix(np.zeros((self.q, self.q)))
        for k in range(self.ng):
            if k != self.fqp.i and k != self.fqp.j:
                if self.fqp.i == -1:
                    b_sum += self.fair_oracle.B[k][self.fqp.j]
                elif self.fqp.j == -1:
                    b_sum += self.fair_oracle.B[k][self.fqp.i]
                else:
                    b_sum += self.fair_oracle.B[k][self.fqp.i]
                    b_sum += self.fair_oracle.B[k][self.fqp.j]

        a_quad = (1.0 - self.fair_oracle.lamb) * self.fair_oracle.a * t_sum
        B_quad = self.fair_oracle.lamb * b_sum
        c_quad = (1.0 - self.fair_oracle.lamb) * (self.fair_oracle.a * self.tr).sum()
        o = Oracle(a_quad, B_quad, c_quad)
        return o


def create_vec_sum_1(l):
    """
    Creates np.array of length n with each item sampled informly from [0, 1] such that
    np.sum(arr) = 1.
    """
    v = np.random.uniform(0, 1, size=l)
    v /= np.sum(v)
    return v


def check_a_B_lamb_T_sphere_satisfy_conditions(nc, ng, a, B, lamb, T, sphere) -> bool:
    """
    Checks if regularity assumption is satisfied for FPME. This essentially runs a loop
    similar to the FPME algorithm but checks that the QME oracle always satisfies regularity
    conditions.
    """
    q = nc ** 2 - nc  # number of off-diagonal elements

    fair_oracle = FairOracle(a, B, lamb, T)
    tr = sphere.origin
    sphere_reparam = Sphere(sphere.origin - tr, sphere.radius, sphere.n)

    fp = FairParametizer(q, ng)
    params = fp.create_paramaterization()

    satisfy = True
    for fqp in params:
        ro = Reparam_QME_FairOracle(fair_oracle, nc, q, ng, fqp, tr)
        ro = ro.convert_to_QME_Oracle()
        if not check_a_B_sphere_satisfy_conditions(sphere_reparam, ro.a, ro.B):
            satisfy = False
            break
    return satisfy


def create_a_B_lamb_T(
    sphere: Sphere, ng: int, nc: int, q: int, well_formed=True, T: List[np.array] = None
):
    """
    Creates a random a, B, lambda, and T such that:

    a: 1D np.array, || a || = 1
    B: 2D list of np.matrix, B[i][j] = B[j][i], 1/2 \sum || B_ij || = 1
    lambda: float, between 0.25 and 0.75
    T: 2D np.array such that np.sum(T, axis=0) = 1

    a represents the linear performance metric.
    B represents the list of discrepancy weights.
    lambda represents the performance/fairness tradeoff.
    T represents the P(G = g | Y = i). T[i] represents the ith group.

    Optionally, T can be provided.
    """
    if T is not None:
        # make sure T is valid
        assert (np.sum(T, axis=0) - 1 < 1e-3).all()

    while True:
        a = normalize(np.random.rand(q))
        while (a < 1e-2).any():
            a = normalize(np.random.rand(q))

        B = [[np.matrix(np.zeros((q, q))) for j in range(ng)] for i in range(ng)]
        unique_bij = ng * (ng - 1) // 2

        # this makes random B_ij such that 1/2 \sum || B_ij || = 1
        B_sum = np.random.uniform(0.0, 1.0, size=unique_bij)
        B_sum = B_sum / np.sum(B_sum) * 2
        idx_B_sum = 0
        for i in range(ng):
            for j in range(i + 1, ng):
                B_ij = np.matrix(make_spd_matrix(q))
                B_ij = normalize(B_ij, B_sum[idx_B_sum])
                B[i][j] = B_ij
                B[j][i] = B_ij
                idx_B_sum += 1

        # keep lambda between 0.25 and 0.75
        lamb = np.random.uniform(0.25, 0.75)

        if T is None:
            # each index c represents the distribution of P(G = g | Y = c)
            # T_sums[c][g] = P(G = g | Y = c), and sum g_i T_sums[c][g_i] = 1
            T_sums = [create_vec_sum_1(ng) for _ in range(nc)]
            T = np.zeros((ng, q))
            for g in range(ng):
                for c in range(nc):
                    # we are flattening the confusion matrix, and len(T[g]) = q
                    # so this copies the value for an entire "row" of the confusion matrix (minus the one off-diag element)
                    T[g, (nc - 1) * c : (nc - 1) * (c + 1)] = T_sums[c][g]

        if not well_formed:
            return a, B, lamb, T

        # return if conditions are satisfied; otherwise, loop
        if check_a_B_lamb_T_sphere_satisfy_conditions(nc, ng, a, B, lamb, T, sphere):
            return a, B, lamb, T

        # reset variables
        T = None


def compute_B_err(B_hat, B_true):
    """
    Compute the error in B. Sums the Frobeneus norm off (B_ij_hat - B_ij) for all i,j such that j > i.
    """
    ng = len(B_hat)
    b_err = 0
    for i in range(ng):
        for j in range(i + 1, ng):
            b_err += np.linalg.norm(B_hat[i][j] - B_true[i][j], ord="fro")
    return b_err
