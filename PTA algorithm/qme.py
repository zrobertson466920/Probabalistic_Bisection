"""
Implementation of the QME algorithm.
"""

from typing import Tuple, List
import numpy as np
from scipy import spatial

from common import MIN_GRAD, SMALL_SPHERE_RADIUS, Sphere, Oracle, normalize
from lpme import LPME


def clip_v(v: np.array):
    """
    Clips elements in v such that | any element | > MIN_GRAD.
    Modifies v in place, and also returns v. Use this in well-formed
    runs of QME because we know gradients satisfy this.
    """
    # bool_mask = v > 0
    # v[(-MIN_GRAD < v) &  ~(bool_mask)] = -MIN_GRAD
    # v[(bool_mask) & (v < MIN_GRAD)] = MIN_GRAD
    if 0 < v[0] < MIN_GRAD:
        v[0] = MIN_GRAD
    elif 0 > v[0] > -MIN_GRAD:
        v[0] = -MIN_GRAD
    return v


class QMESC:
    def __init__(
        self,
        sphere: Sphere,
        rS: float,
        f_z: np.array,
        f_neg0: np.array,
        fs: List[np.array],
        well_formed: bool,
    ):
        """
        QMESC: QME Slope Calculator class that can calculate (a, B) given estimate of
        normalized gradient at specific points. This info is acquired by queries to LPME in
        the QME algorithm. That is then given to this class, which uses its methods to recover (a, B).

        Initialize QMESC with estimates of normalized gradients at specific points, explained below:

        sphere: Sphere that represents query space
        rS: radius of small sphere (used in LPME algorithm)
        f_z: estimated slope at origin (o) of search space
        f_neg0: estimated slope at: o - (rB - rS) * <1, 0, ...., 0>
        fs: for each index i, fs_i = est. slope at: o + (rB - rS) * <0, ..., 1 (at ith position), ..., 0>
        well_formed: whether true gradients are well-formed
        """
        self.sphere = sphere
        self.rS = rS
        self.f_z = f_z
        self.f_neg0 = f_neg0
        self.fs = fs
        self.well_formed = well_formed

        self.cmn = None  # will be set during the first call to compute_bij and reused
        self.d0 = None  # computed using norms

    def compute_bij(self, i: int, j: int):
        """
        Recovers the value at b_ij in terms of d0, the true (unnormalized) first coordinate of d.
        See paper for an explanation of d.

        i = row
        j = col

        b_ij = c_ij * d0
        Returns c_ij
        """
        f_i_j = self.fs[j][i]
        f_0_j = self.fs[j][0]
        f_j_0 = self.fs[0][j]
        f_0_0 = self.fs[0][0]
        f_1_0 = self.fs[0][1]

        f_0_z = self.f_z[0]
        f_1_z = self.f_z[1]
        f_j_z = self.f_z[j]
        f_i_z = self.f_z[i]

        f_1_neg0 = self.f_neg0[1]
        f_0_neg0 = self.f_neg0[0]

        # this is used by all computations of B_ij
        if self.cmn is None:
            # in the well_formed case, abs(denom) > MIN_GRAD
            denom = f_1_neg0 / f_0_neg0 - f_1_0 / f_0_0
            if self.well_formed:
                if 0 < denom < MIN_GRAD:
                    # clip, must be > MIN_GRAD
                    denom = MIN_GRAD
                elif -MIN_GRAD < denom < 0:
                    # clip, must be < MIN_HRAD
                    denom = -MIN_GRAD

            self.cmn = (
                (f_1_neg0 / f_0_neg0) + (f_1_0 / f_0_0) - 2 * (f_1_z / f_0_z)
            ) / denom

        # see paper Eq. 12
        exp = (
            (f_i_j / f_0_j) * (1 + f_j_0 / f_0_0)
            - (f_i_j * f_j_z) / (f_0_j * f_0_z)
            - (f_i_z / f_0_z)
            + ((f_i_j * f_j_0) / (f_0_j * f_0_0)) * self.cmn
        )

        return exp / (self.sphere.radius - self.rS)

    def compute_di(self, i: int):
        """
        Recovers the value at d_i in terms of d0, the true (unnormalized) first coordinate of d.
        See paper for an explanation of d.

        i = index

        d_i = c_i * d0
        Returns c_i
        """
        return self.f_z[i] / self.f_z[0]

    def compute_a_b(self) -> Tuple[np.array, np.matrix]:
        """
        Recovers (a, B) using the equations in the paper. Uses compute_bij and compute_di
        to get (a, B) in terms of d0. Then, since || a ||_2^2 + || B ||_2^2 = 1, d0 can be recovered.
        After d0 is recovered, (a, B) can be recovered.

        Returns (a, B)
        """
        bhat = np.matrix(np.zeros((self.sphere.n, self.sphere.n)))
        for i in range(0, self.sphere.n):
            for j in range(i, self.sphere.n):
                bhat[i, j] = self.compute_bij(i, j)
                bhat[j, i] = bhat[i, j]

        dhat = np.zeros(self.sphere.n)
        for i in range(0, self.sphere.n):
            dhat[i] = self.compute_di(i)

        ahat = dhat - np.array(bhat @ self.sphere.origin)[0]

        # |d0| * (a_norm + b_norm) = 1.0
        b_norm = np.sum(np.square(bhat))
        a_norm = np.sum(np.square(ahat))
        self.d0 = 1.0 / np.sqrt(a_norm + b_norm)
        # if sign is negative, use the negative solution
        if self.f_z[0] < 0:
            self.d0 = -self.d0
        return ahat * self.d0, bhat * self.d0


class QME:
    def __init__(
        self,
        sphere: Sphere,
        oracle: Oracle,
        e: float,
        well_formed: bool = True,
        check_i: bool = False,
    ) -> Tuple[np.array, np.matrix]:
        """
        Quadratic Metric Elicitation algorithm. Recovers the quadratic metric
        hidden by the oracle with spherical search space using only pairwise comparisons.
        Assumes metric of the form:

        score = <a,r> + 0.5 r^T B r

        such that || a || + 0.5 || B || = 1

        Recovers (a, B)

        Queries are only inside the search sphere. Note that a search sphere centered at the origin
        will have higher practical performance, because the first derivative d = a + Bo.
        B is harder to recover in practice due to fractional errors (see qme.ipynb). When o = 0, d = a.
        a is easier to recover, making d more accurate. d directly affects the predicted a, B
        (see paper), so a more accurate d improves a, B. Also, realize that any true search space
        S can be transformed into one that is centered at the origin without changing the metric.
        See Reparam_QME_FairOracle and FPME in fpme.py for an example.

        sphere: search space
        oracle: oracle with hidden metric
        e: search tolerance
        well_formed: are (a, B) well formed, meaning abs(gradient element) > MIN_GRAD
        check_i: boolean indicating whether to check for inconsistencies
        """
        self.sphere = sphere
        self.oracle = oracle
        self.e = e
        self.well_formed = well_formed
        self.check_i = check_i

        # after calling run_qme this will be saved
        self.qmesc = None

        if self.check_i:
            # computes the true d
            self.d = self.oracle.a + np.array(self.oracle.B @ sphere.origin)[0]
            self.inconsistencies = list()

    def run_qme(self, alpha) -> Tuple[np.array, np.matrix]:
        """
        Runs QME algorithm. Elicts hidden a, B just using oracle comparions.
        Uses LPME as a subroutine. See paper and code for details.
        """
        q = self.sphere.n
        rB = self.sphere.radius  # radius of large sphere
        rS = SMALL_SPHERE_RADIUS  # radius of small sphere

        f_z_small_sphere = Sphere(self.sphere.origin, rS, q)
        lpm_z = LPME(f_z_small_sphere, self.oracle, self.e, check_i=self.check_i)
        f_z = lpm_z.run_lpme(alpha)
        if self.well_formed:
            f_z = clip_v(f_z)
        self.lpm_z = lpm_z

        if self.check_i:
            self.f_z_opt = self.d
            self.check_inconsistency(self.f_z_opt, f_z, "f_z")
            for incon in self.lpm_z.inconsistencies:
                self.inconsistencies.append(f"lpm_fz: {incon}")

        # z points in the - direction in the first dimension
        z_neg0 = np.zeros(q)
        z_neg0[0] = -(rB - rS)
        f_neg0_small_sphere = Sphere(self.sphere.origin + z_neg0, rS, q)
        lpm_neg0 = LPME(f_neg0_small_sphere, self.oracle, self.e, check_i=self.check_i)
        f_neg0 = lpm_neg0.run_lpme(alpha)
        if self.well_formed:
            f_neg0 = clip_v(f_neg0)
        self.lpm_neg0 = lpm_neg0

        if self.check_i:
            self.f_neg0_opt = self.d + np.array(self.oracle.B @ z_neg0)[0]
            self.check_inconsistency(self.f_neg0_opt, f_neg0, "f_neg0")
            for incon in self.lpm_neg0.inconsistencies:
                self.inconsistencies.append(f"lpm_f_neg0: {incon}")

        fs = list()
        self.lpms = list()
        if self.check_i:
            self.fs_opt = list()
        for i in range(0, q):
            z_i = np.zeros(q)
            z_i[i] = rB - rS
            f_i_small_sphere = Sphere(self.sphere.origin + z_i, rS, q)
            lpm_i = LPME(f_i_small_sphere, self.oracle, self.e, check_i=self.check_i)
            f_i = lpm_i.run_lpme(alpha)
            if self.check_i:
                f_i_opt = self.d + np.array(self.oracle.B @ z_i)[0]
                self.check_inconsistency(f_i_opt, f_i, f"f_{i}")
                self.fs_opt.append(f_i_opt)
                for incon in lpm_i.inconsistencies:
                    self.inconsistencies.append(f"lpm_f_{i}: {incon}")

            if self.well_formed:
                f_i = clip_v(f_i)
            self.lpms.append(lpm_i)
            fs.append(f_i)

        self.qmesc = QMESC(
            self.sphere, rS, f_z, f_neg0, fs, well_formed=self.well_formed
        )
        return self.qmesc.compute_a_b()

    def run_qme_opt(self):
        """
        Computes and uses the true gradients instead of estimating with oracle queries.
        The state is saved to self.qmesc_opt and does not interfere with the state
        saved by run_qme. So, this can be run before/after a call to run_qme.
        """
        q = self.sphere.n
        rB = self.sphere.radius  # radius of large sphere
        rS = SMALL_SPHERE_RADIUS  # radius of small sphere

        d = self.oracle.a + np.array(self.oracle.B @ self.sphere.origin)[0]
        f_z_opt = d

        # z points in the - direction in the first dimension
        z_neg0 = np.zeros(q)
        z_neg0[0] = -(rB - rS)
        f_neg0_opt = d + np.array(self.oracle.B @ z_neg0)[0]

        fs_opt = list()
        for i in range(0, q):
            z_i = np.zeros(q)
            z_i[i] = rB - rS
            f_i_opt = d + np.array(self.oracle.B @ z_i)[0]
            fs_opt.append(f_i_opt)

        # set to not well-formed because we want to use inputs directly
        self.qmesc_opt = QMESC(
            self.sphere, rS, f_z_opt, f_neg0_opt, fs_opt, well_formed=False
        )
        return self.qmesc_opt.compute_a_b()

    def check_inconsistency(
        self, grad_expected: np.array, grad_actual: np.array, context: str
    ):
        """
        Computes the cosine similarity between grad_expected and grad_actual.
        If the radians distance is more than 1e-3, it is called an "inconsistency" and stored in a list.
        The context should be a string that uniquely identifies the gradients.
        """
        dist = spatial.distance.cosine(grad_expected, grad_actual)
        if dist > 1e-3:
            # 5.7 degrees off
            self.inconsistencies.append(context)
