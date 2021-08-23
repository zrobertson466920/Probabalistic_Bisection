"""
Implementation of the LPME algorithm.
"""

import numpy as np
import math
from typing import List
from simpleDist import Simple_Dist

from common import Oracle, Sphere, ThetaSearchSpace, normalize


def initialize_orthants(signs: List[int]):
    """
    signs: list of signs in n-d space indicating the known sign of that dimension

    Returns a list of (n-1) search spaces that defines the bounds for spherical coordinate
    angles that obey the signs when converted to cartesian coordinates
    """
    orthants = []
    for i in range(0, len(signs) - 2):
        if signs[i] > 0:
            orthants.append(ThetaSearchSpace(0.0, math.pi / 2))
        else:
            orthants.append(ThetaSearchSpace(math.pi / 2, math.pi))

    # choose last theta
    if signs[-2] < 0 and signs[-1] < 0:
        orthants.append(ThetaSearchSpace(math.pi, 3 * math.pi / 2))
    elif signs[-2] < 0 and signs[-1] > 0:
        orthants.append(ThetaSearchSpace(math.pi / 2, math.pi))
    elif signs[-2] > 0 and signs[-1] < 0:
        orthants.append(ThetaSearchSpace(3 * math.pi / 2, 2 * math.pi))
    else:
        orthants.append(ThetaSearchSpace(0.0, math.pi / 2))

    return orthants


def compute_vector(sphere: Sphere, theta_list: List[float]) -> np.array:
    """
    theta_list: list of spherical coordinate angles

    Converts into vector and projects onto sphere. Used to query in search space.
    """
    vec = []
    x0 = math.cos(theta_list[0])
    vec.append(x0)

    cur_sin_product = math.sin(theta_list[0])
    for theta in theta_list[1:]:
        xi = cur_sin_product * math.cos(theta)
        vec.append(xi)
        cur_sin_product *= math.sin(theta)
    # last one all sines
    vec.append(cur_sin_product)
    vec = np.array(vec)
    vec = vec * sphere.radius + sphere.origin
    return vec


class LPME:
    def __init__(self, sphere: Sphere, oracle: Oracle, e: float, check_i: bool = False):
        """
        Linear Performance Metric Elicitation algorithm. Recovers the linear metric
        hidden by the oracle with spherical search space using only pairwise comparisons.
        Assumes metric of the form:

        score = <a, r>

        Recovers a such that || a || = 1

        Queries are only inside the search sphere

        sphere: search space
        oracle: oracle with hidden metric
        e: search tolerance
        check_i: a boolean indicating whether inconsistencies should be checked.
            See self.check_inconsistency for more info
        """
        self.sphere = sphere
        self.oracle = oracle
        self.e = e
        self.check_i = check_i

        # these are computed and saved after run_lpme is called
        self.theta_list = None
        self.num_queries = 0

        if self.check_i:
            self.inconsistencies = list()

    def run_lpme(self, alpha:float) -> np.array:
        """
        Runs LPME algorithm. Elicits a using just oracle comparisons.

        Effectively works binary-search style on the (spherical) search space.
        See paper for details.
        """
        q = self.sphere.n
        signs = []
        for i in range(q):
            a = np.ones(q)
            a = a / np.sqrt(q)
            a_prime = np.copy(a)
            a_prime[i] = -a_prime[i]

            z_a = a * self.sphere.radius + self.sphere.origin
            z_a_prime = a_prime * self.sphere.radius + self.sphere.origin

            if self.oracle.compare(z_a, z_a_prime, alpha):
                signs.append(1.0)
            else:
                signs.append(-1.0)

        orthants = initialize_orthants(signs)

        # number of cycles
        nc = 4
        theta_list = [(orth.start + orth.stop) / 2 for orth in orthants]
        for _ in range(0, nc):
            for j in range(0, q - 1):
                theta_a = orthants[j].start
                theta_b = orthants[j].stop
                dist = Simple_Dist(init=[(theta_a,1/(theta_b-theta_a))],end=theta_b)
                # while abs(theta_b - theta_a) > self.e:
                for iter in range(100):
                    # dist.display()

                    theta_c = dist.invcdf(1/3)
                    theta_d = dist.invcdf(2/3)

                    theta_list[j] = theta_c
                    vec_c = compute_vector(self.sphere, theta_list)
                    
                    theta_list[j] = theta_d
                    vec_d = compute_vector(self.sphere, theta_list)
                    # print(vec_c, vec_d)


                    # query
                    if not self.oracle.compare(vec_c, vec_d, alpha):
                        dist.update_dist_interval(left=theta_c, right=theta_b, vote=1, alpha=alpha)
                    else:
                        dist.update_dist_interval(left=theta_a, right=theta_d, vote=1, alpha=alpha)
                    self.num_queries += 1

                # update theta list
                theta_list[j] = dist.median()

        # save theta list
        self.theta_list = theta_list
        return normalize(compute_vector(self.sphere, theta_list) - self.sphere.origin) 
    
    def run_lpme_nPTA(self, alpha:float) -> np.array:
        """
        Runs LPME algorithm. Elicits a using just oracle comparisons.

        Effectively works binary-search style on the (spherical) search space.
        See paper for details.
        """
        q = self.sphere.n
        signs = []
        for i in range(q):
            a = np.ones(q)
            a = a / np.sqrt(q)
            a_prime = np.copy(a)
            a_prime[i] = -a_prime[i]

            z_a = a * self.sphere.radius + self.sphere.origin
            z_a_prime = a_prime * self.sphere.radius + self.sphere.origin

            if self.oracle.compare(z_a, z_a_prime, alpha):
                signs.append(1.0)
            else:
                signs.append(-1.0)

        orthants = initialize_orthants(signs)

        # number of cycles
        nc = 4
        theta_list = [(orth.start + orth.stop) / 2 for orth in orthants]
        for _ in range(0, nc):
            for j in range(0, q - 1):
                theta_a = orthants[j].start
                theta_b = orthants[j].stop
                while abs(theta_b - theta_a) > self.e:
                    theta_c = (theta_a * 3 + theta_b) / 4
                    theta_d = (theta_a + theta_b) / 2
                    theta_e = (theta_a + theta_b * 3) / 4

                    theta_list[j] = theta_a
                    vec_a = compute_vector(self.sphere, theta_list)

                    theta_list[j] = theta_b
                    vec_b = compute_vector(self.sphere, theta_list)

                    theta_list[j] = theta_c
                    vec_c = compute_vector(self.sphere, theta_list)

                    theta_list[j] = theta_d
                    vec_d = compute_vector(self.sphere, theta_list)

                    theta_list[j] = theta_e
                    vec_e = compute_vector(self.sphere, theta_list)

                    # compare ac
                    cac = self.oracle.compare(vec_a, vec_c, alpha)
                    ccd = self.oracle.compare(vec_c, vec_d, alpha)
                    cde = self.oracle.compare(vec_d, vec_e, alpha)
                    ceb = self.oracle.compare(vec_e, vec_b, alpha)
                    self.num_queries += 4

                    if self.check_i:
                        context = {
                            "theta_list": theta_list,
                            "j": j,
                            "theta_a": theta_a,
                            "theta_b": theta_b,
                            "theta_c": theta_c,
                            "theta_d": theta_d,
                            "theta_e": theta_e,
                        }
                        self.check_inconsistency(cac, ccd, cde, ceb, context)

                    if cac:
                        theta_b = theta_d
                    elif ccd:
                        theta_b = theta_d
                    elif cde:
                        theta_a = theta_c
                        theta_b = theta_e
                    elif ceb:
                        theta_a = theta_d
                    else:
                        theta_a = theta_d

                # update theta list
                theta_list[j] = (theta_a + theta_b) / 2

        # save theta list
        self.theta_list = theta_list
        return normalize(compute_vector(self.sphere, theta_list) - self.sphere.origin)

    def check_inconsistency(
        self, cac: bool, ccd: bool, cde: bool, ceb: bool, context: dict
    ):
        """
        Given oracle comparisons, checks to see if any are inconsistent with
        true (theoretical) linear optimization on a sphere.

        cac: bool, comparison vec_a > vec_c
        ccd: bool, comparison between vec_c > vec_d
        cde: bool, comparison between vec_d > vec_e
        ceb: bool, comparison vec_e > vec_b
        context: dict containing
            - theta_list: list, current theta_list
            - j: int, current index into theta_list
            - theta_a: float, produced vec_a
            - theta_b: float, produced vec_b
            - theta_c: float, produced vec_c
            - theta_d: float, produced vec_d
            - theta_e: float, produced vec_e


        Adds inconsistencies to list
        """
        # if vec_a > vec_c, then vec_a should be greater than everyone else
        if cac:
            if not ccd:
                # if vec_a > vec_c then vec_a should be greater than everyone else
                i_info = (context, "a > c but c < d")
                self.inconsistencies.append(i_info)
            if not cde:
                i_info = (context, "a > c but d < e")
                self.inconsistencies.append(i_info)
            if not ceb:
                i_info = (context, "a > c but e < b")
                self.inconsistencies.append(i_info)

        # vec_a < vec_c and vec_c > vec_d, then vec_c should be greater than everyone else
        if ccd:
            if not cde:
                i_info = (context, "a < c, c > d but d < e")
                self.inconsistencies.append(i_info)
            if not ceb:
                i_info = (context, "a < c, c > d but e < b")
                self.inconsistencies.append(i_info)

        # vec_a < vec_c, vec_c < vec_d, vec_d > vec_e, then vec_c should be greater than everyone else
        if cde:
            if not ceb:
                i_info = (context, "a < c, c < d, d > e but e < b")
                self.inconsistencies.append(i_info)
