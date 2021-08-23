"""
Defines functions for generating trials for FPME/QME and saving trial results.
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import pickle
from typing import List

import sys

sys.path.append("../")
from common import Sphere, Oracle, create_a_B, normalize
from fpme_utils import create_a_B_lamb_T, compute_B_err
from fpme import FPME, FairOracle, Reparam_QME_FairOracle
from qme import QME


NUM_TRIALS = 300


## --- QME ---


def create_qme_trial(nc: int, q: int, trials: int, well_formed: bool):
    """
    well_formed: whether the trial should contain well_formed (a, B)
    Create `trials` trials for `nc` classes. Saves to folder trials/qme/k=nc.
    """
    np.random.seed(7)  # reproducability

    os.makedirs("trials", exist_ok=True)
    os.makedirs("trials/qme", exist_ok=True)
    os.makedirs("trials/qme/well_formed", exist_ok=True)
    os.makedirs("trials/qme/ill_formed", exist_ok=True)
    bp = None
    if well_formed:
        bp = "trials/qme/well_formed/k={}".format(nc)
    else:
        bp = "trials/qme/ill_formed/k={}".format(nc)
    if os.path.exists(bp):
        print("k={} exists".format(nc))
        return
    os.makedirs(bp, exist_ok=False)

    sphere = Sphere(np.zeros(q), 1.0, q)
    f = open(os.path.join(bp, "sphere.pkl"), "wb")
    pickle.dump(sphere, f)
    f.close()

    for i in tqdm(range(trials)):
        a, B = create_a_B(sphere, q, well_formed=well_formed)

        np.save(os.path.join(bp, "a_{}.npy".format(i)), a)

        f = open(os.path.join(bp, "B_{}.pkl".format(i)), "wb")
        pickle.dump(B, f)
        f.close()


def load_qme_sphere(nc: int, well_formed: bool):
    """
    Load the fpme sphere for some `nc` and trials with `well_formed` inputs
    """
    bp = None
    if well_formed:
        bp = "trials/qme/well_formed/k={}".format(nc)
    else:
        bp = "trials/qme/ill_formed/k={}".format(nc)

    f = open(os.path.join(bp, "sphere.pkl"), "rb")
    sphere = pickle.load(f)
    f.close()
    return sphere


def load_a_B(nc: int, i: int, well_formed: bool):
    """
    Load the a, B for some `nc` classes and trial `i` with `well_formed` inputs
    """
    bp = None
    if well_formed:
        bp = "trials/qme/well_formed/k={}".format(nc)
    else:
        bp = "trials/qme/ill_formed/k={}".format(nc)

    a = np.load(os.path.join(bp, "a_{}.npy".format(i)))

    f = open(os.path.join(bp, "B_{}.pkl".format(i)), "rb")
    B = pickle.load(f)
    f.close()

    return a, B


def load_a_B_hat(nc: int, i: int, well_formed: bool):
    """
    Load the predicted a, B for some `nc` classes and trial `i` with `well_formed` inputs
    """
    bp = None
    if well_formed:
        bp = "trials/qme/well_formed/k={}".format(nc)
    else:
        bp = "trials/qme/ill_formed/k={}".format(nc)

    a = np.load(os.path.join(bp, "a_{}_hat.npy".format(i)))

    f = open(os.path.join(bp, "B_{}_hat.pkl".format(i)), "rb")
    B = pickle.load(f)
    f.close()

    return a, B


def write_qme_trial(
    nc: int,
    i: int,
    a_hat: np.array,
    B_hat: np.matrix,
    well_formed: bool,
):
    """
    Save the `a_hat`, `B_hat` for some `nc` and trial `i` with `well_formed` inputs
    """
    bp = None
    if well_formed:
        bp = "trials/qme/well_formed/k={}".format(nc)
    else:
        bp = "trials/qme/ill_formed/k={}".format(nc)
    if not os.path.exists(bp):
        print("k={} does not exist".format(nc))
        return

    np.save(os.path.join(bp, "a_{}_hat.npy".format(i)), a_hat)

    f = open(os.path.join(bp, "B_{}_hat.pkl".format(i)), "wb")
    pickle.dump(B_hat, f)
    f.close()


def write_qme_trial_summary(
    nc: int, a_err: np.array, B_err: np.array, well_formed: bool
):
    """
    Save the errors in a_hats, B_hats for all trials on `nc` classes with `well_formed` inputs
    """
    assert len(a_err) == NUM_TRIALS
    assert a_err.shape == B_err.shape
    bp = None
    if well_formed:
        bp = "trials/qme/well_formed/k={}".format(nc)
    else:
        bp = "trials/qme/ill_formed/k={}".format(nc)
    if not os.path.exists(bp):
        print("k={} does not exist".format(nc))
        return

    np.save(os.path.join(bp, "a_err.npy"), a_err)
    np.save(os.path.join(bp, "B_err.npy"), B_err)


def load_qme_trial_summary(nc: int, well_formed: bool):
    """
    Loads the errors in a_hats, B_hats for all trials on `nc` classes with `well_formed` inputs
    """
    bp = None
    if well_formed:
        bp = "trials/qme/well_formed/k={}".format(nc)
    else:
        bp = "trials/qme/ill_formed/k={}".format(nc)
    if not os.path.exists(bp):
        print("k={} does not exist".format(nc))
        return

    a_err = np.load(os.path.join(bp, "a_err.npy"))
    B_err = np.load(os.path.join(bp, "B_err.npy"))
    return a_err, B_err


## --- FPME ---


def create_fpme_trial(ng: int, nc: int, q: int, trials: int):
    """
    Create `trials` instances of fpme problem with random (sphere, a, B, lamb, T) for the provided (ng, nc).
    Saves to folder trials/qme/m=ng,k=nc
    """
    np.random.seed(7)  # reproducability

    os.makedirs("trials", exist_ok=True)
    os.makedirs("trials/fpme", exist_ok=True)
    bp = "trials/fpme/m={},k={}".format(ng, nc)
    if os.path.exists(bp):
        print("m={},k={} exists".format(ng, nc))
        return
    os.makedirs(bp, exist_ok=False)

    sphere = Sphere(np.random.randn(q), 1.0, q)
    f = open(os.path.join(bp, "sphere.pkl"), "wb")
    pickle.dump(sphere, f)
    f.close()

    for i in tqdm(range(trials)):
        a, B, lamb, T = create_a_B_lamb_T(sphere, ng, nc, q, well_formed=True)

        np.save(os.path.join(bp, "a_{}.npy".format(i)), a)

        f = open(os.path.join(bp, "B_{}.pkl".format(i)), "wb")
        pickle.dump(B, f)
        f.close()

        f = open(os.path.join(bp, "lamb_{}.pkl".format(i)), "wb")
        pickle.dump(lamb, f)
        f.close()

        np.save(os.path.join(bp, "T_{}.npy".format(i)), T)


def load_fpme_sphere(ng: int, nc: int):
    """
    Load the fpme sphere for some `ng`, `nc`
    """
    bp = "trials/fpme/m={},k={}".format(ng, nc)
    f = open(os.path.join(bp, "sphere.pkl"), "rb")
    sphere = pickle.load(f)
    f.close()
    return sphere


def load_a_B_lamb_T(ng: int, nc: int, i: int):
    """
    Load the a, B, lamb, T for some `ng`, `nc`, and trial `i`
    """
    bp = "trials/fpme/m={},k={}".format(ng, nc)
    a = np.load(os.path.join(bp, "a_{}.npy".format(i)))

    f = open(os.path.join(bp, "B_{}.pkl".format(i)), "rb")
    B = pickle.load(f)
    f.close()

    f = open(os.path.join(bp, "lamb_{}.pkl".format(i)), "rb")
    lamb = pickle.load(f)
    f.close()

    T = np.load(os.path.join(bp, "T_{}.npy".format(i)))

    return a, B, lamb, T


def write_fpme_trial(
    ng: int,
    nc: int,
    i: int,
    a_hat: np.array,
    B_hat: List[List[np.matrix]],
    lamb_hat: float,
):
    """
    Save the a_hat, B_hat, lamb_hat for some `ng`, `nc`, and trial `i`
    """
    bp = "trials/fpme/m={},k={}".format(ng, nc)
    if not os.path.exists(bp):
        print("m={},k={} does not exist".format(ng, nc))
        return

    np.save(os.path.join(bp, "a_{}_hat.npy".format(i)), a_hat)

    f = open(os.path.join(bp, "B_{}_hat.pkl".format(i)), "wb")
    pickle.dump(B_hat, f)
    f.close()

    f = open(os.path.join(bp, "lamb_{}_hat.pkl".format(i)), "wb")
    pickle.dump(lamb_hat, f)
    f.close()


def write_fpme_trial_summary(
    ng: int, nc: int, a_err: np.array, B_err: np.array, lamb_err: float
):
    """
    Save the errors in a_hats, B_hats, lamb_hats for a trial.

    `a_err`
    """
    assert len(a_err) == NUM_TRIALS
    assert a_err.shape == B_err.shape == lamb_err.shape
    bp = "trials/fpme/m={},k={}".format(ng, nc)
    if not os.path.exists(bp):
        print("m={},k={} does not exist".format(ng, nc))
        return

    np.save(os.path.join(bp, "a_err.npy"), a_err)
    np.save(os.path.join(bp, "B_err.npy"), B_err)
    np.save(os.path.join(bp, "lamb_err.npy"), lamb_err)


def load_fpme_trial_summary(ng: int, nc: int):
    """
    Loads the errors in a_hats, B_hats, lamb_hats for all trials on `ng` groups, `nc` classes
    """
    bp = "trials/fpme/m={},k={}".format(ng, nc)
    if not os.path.exists(bp):
        print("k={} does not exist".format(nc))
        return

    a_err = np.load(os.path.join(bp, "a_err.npy"))
    B_err = np.load(os.path.join(bp, "B_err.npy"))
    lamb_err = np.load(os.path.join(bp, "lamb_err.npy"))
    return a_err, B_err, lamb_err


## --- Main ---


if __name__ == "__main__":
    print("generating fpme trials")
    for ng in range(2, 6):
        for nc in range(2, 6):
            q = nc ** 2 - nc
            create_fpme_trial(ng, nc, q, NUM_TRIALS)

    print("generating well-formed qme trials")
    for nc in range(2, 6):
        q = nc ** 2 - nc
        create_qme_trial(nc, q, NUM_TRIALS, well_formed=True)

    print("generating ill-formed qme trials")
    for nc in range(2, 6):
        q = nc ** 2 - nc
        create_qme_trial(nc, q, NUM_TRIALS, well_formed=False)
