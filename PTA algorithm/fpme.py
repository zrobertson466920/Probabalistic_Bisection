"""
Implementation of the FPME algorithm.
"""

from typing import List, Tuple
import numpy as np
from scipy import spatial

from common import Sphere
from fpme_utils import (
    FairOracle,
    Reparam_LPME_FairOracle,
    Reparam_QME_FairOracle,
    FairParametizer,
    Fair_QME_Parameterization,
)
from lpme import LPME
from qme import QME


class FPME:
    """
    Fair Performance Metric Elicitation algorithm. Recovers the fair metric
    hidden by the oracle with spherical search space using only pairwise comparisons.
    """

    def __init__(
        self,
        sphere: Sphere,
        fair_oracle: FairOracle,
        T: np.array,
        nc: int,
        q: int,
        ng: int,
        e: float,
        well_formed: bool = True,
        check_i: bool = False,
    ):
        """
        sphere: search space
        fair_oracle: fair_oracle with hidden metric
        T: 2D np.array, represents the P(G = g | Y = i). T[i] represents the ith group. See fpme_utils.py:FairOracle
            for how this is used. See fpme.ipynb for an example.
        nc: number of classes
        q: number of off-diag rate elements, should be nc ** 2 - nc
        ng: number of groups
        e: search tolerance
        well_formed: are (a, B, lamb, T) well formed, meaning abs(gradient element) > MIN_GRAD for all invocations of QME.
            See fpme_utils.py:create_a_B_lamb_T for how this can be guaranteed during generation.
        check_i: check for inconsistencies in gradient estimations, looks at Oracle to do so
        """
        self.sphere = sphere
        self.fair_oracle = fair_oracle
        self.T = T
        self.nc = nc
        self.q = q
        self.ng = ng
        self.e = e
        self.well_formed = well_formed
        self.check_i = check_i

        # the trivial rate will be the sphere's origin
        self.tr = sphere.origin

        # a class that helps enumerate paramaterizations and then
        # solve for measurements associated with paramaterizations
        self.fp = FairParametizer(self.q, self.ng)

        # enumerates all paramterizations to help elicit B
        self.params = self.fp.create_paramaterization()

        # sphere S is originally defined so all s \in S
        # reparameterize sphere as it searches in in f space, where:
        # f = s - tr
        self.sphere_reparam = Sphere(
            self.sphere.origin - self.tr, self.sphere.radius, self.sphere.n
        )
        self.qms = list()
        self.b_measurements = list()
        self.a_list = list()

        # stores inconsistencies discovered
        if self.check_i:
            self.inconsistencies = list()

    def run_fpme(
        self, solve_opt_qme: bool = False, alpha:float=0
    ) -> Tuple[np.array, List[List[np.matrix]], float]:
        """
        Run FPME algorithm. Elicits hidden a, B, lambda using just oracle comparons.

        Uses LPME and QME as subroutines. See paper and code for details.
        """
        # reparameterize the fair oracle so that LPME thinks it is using
        # an LPME oracle but it is being mapped to the fair oracle
        # the parameterization is simply <s> -> [s, s, ....] (for all groups)
        ro = Reparam_LPME_FairOracle(self.fair_oracle, self.ng)
        if self.check_i:
            ro = ro.convert_to_LPME_Oracle()
        self.lpm = LPME(self.sphere, ro, self.e)
        ahat = self.lpm.run_lpme(alpha)

        # now elicit B, lamb
        for i, p in enumerate(self.params):
            # reparameterize the fair oracle so that QME thinks it is using
            # a QME oracle but it is being mapped to the fair oracle as defined by parameterization p
            ro = Reparam_QME_FairOracle(
                self.fair_oracle, self.nc, self.q, self.ng, p, self.tr
            )

            if self.check_i or solve_opt_qme:
                # when checking for inconsistencies, convert to a true QME oracle
                # so that QME can check for inconsistencies properly
                ro = ro.convert_to_QME_Oracle()

            qm = QME(
                self.sphere_reparam,
                ro,
                self.e,
                well_formed=self.well_formed,
                check_i=self.check_i,
            )
            ahat_qme, bhat_qme = None, None
            if solve_opt_qme:
                ahat_qme, bhat_qme = qm.run_qme_opt(alpha)
            else:
                ahat_qme, bhat_qme = qm.run_qme(alpha)

            t_sum = np.zeros(ahat.shape)
            if p.i != -1:
                t_sum += self.T[p.i]
            if p.j != -1:
                t_sum += self.T[p.j]
            # a_m_hat = (1 - lamb) * a_true * (T[p.i] + T[p.j])
            a_m_hat = ahat * t_sum  # * (1 - lamb), but that is unknown

            if self.check_i:
                # a_m_hat and ahat_qme should lie on the same image
                if spatial.distance.cosine(a_m_hat, ahat_qme) > 1e-3:
                    self.inconsistencies.append(
                        f"a_m_hat and ahat_qme misaligned on parameterization {i}"
                    )
                for incon in qm.inconsistencies:
                    self.inconsistencies.append(
                        f"qm inconsistency on parameterization {i}: {incon}"
                    )

            # this scales B to be just in terms of lamb / (1 - lamb)
            # note that having sphere_reparam centered on (0,0,...) helps as
            # d0 = a0, so there is no reliance on B. We can just divide out ahat_qme[0] and
            # then multiply by a_m_hat to scale the result
            # we take the mean here because that gives us a better estimate of the ratio between
            # a_m_hat and ahat_qme
            b_measured = bhat_qme * np.mean(a_m_hat / ahat_qme)
            self.b_measurements.append(b_measured)
            self.qms.append(qm)
            self.a_list.append(ahat_qme)

        Bhat, lamb_hat = self.fp.recover_B_lambda_from_measurements(
            self.b_measurements, self.params
        )

        return ahat, Bhat, lamb_hat
