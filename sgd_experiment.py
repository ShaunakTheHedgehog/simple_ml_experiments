"""Streaming SGD for matrix linear regression, with an ODE prediction for the risk.

Setup
-----
True weights  W*  in R^{d x d}  with  (W*)_ij ~ N(0, 1/d^2)   (so ||W*||_F = O(1)).
We learn      W   in R^{d x d}  by streaming isotropic inputs x_t ~ N(0, I_d) one at
a time and applying the SGD update

        W(t+1) = W(t) - (gamma_tilde / d) * x_t x_t^T (W(t) - W*) ,   gamma_tilde = 1.

Initialization  (W(0))_ij ~ N(0, 1/d^2)  so ||W(0)||_F = O(1) in expectation.

Prediction risk:   R(W) = 1/2 ||W - W*||_F^2.

For each d we fix one W* and one W(0), then run T = 20*d streaming-SGD steps,
repeating 10 times with the SAME W* and W(0) (only the streamed inputs differ),
and report the mean +/- std of the risk curve.

ODE prediction
--------------
With T_ODE = T/d = 20 and rho(0) = R(0) = 1/2 ||W(0) - W*||_F^2,

        d/dt rho = (-gamma_tilde + gamma_tilde^2 / 2) rho ,

solved by Euler's method on t in [0, T_ODE]. The risk at SGD step k is rho(k/d),
so plotting rho against t lines up with the empirical x-axis (iterations / d).
Because W* and W(0) are fixed per d, there is a SINGLE theory curve per d.
"""

import functools
import os

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_enable_x64", True)  # risk decays many orders of magnitude

# ----------------------------- experiment config -----------------------------
DS = [100, 200, 400, 800, 1600]
N_RUNS = 10
GAMMA_TILDE = 1.0
T_FACTOR = 20          # T = T_FACTOR * d
ODE_DT = 1e-3          # Euler step in the rescaled time t = (sgd step) / d
SEED = 0
OUT_PNG = "sgd_risk.png"
CACHE_NPZ = "sgd_curves.npz"  # cached empirical risk curves (SGD is the slow part)


def risk(E):
    """Prediction risk R = 1/2 ||E||_F^2 for E = W - W*."""
    return 0.5 * jnp.sum(E * E)


@functools.partial(jax.jit, static_argnums=(2, 3))
def run_sgd(E0, stream_key, n_steps, d, gamma_tilde):
    """One streaming-SGD trajectory. Returns risks at steps 0..n_steps (len n_steps+1)."""
    coef = gamma_tilde / d

    def step(E, key):
        x = jax.random.normal(key, (d,), dtype=E.dtype)
        # x x^T E = x (x^T E):  update W(t+1) = W(t) - (gamma/d) x x^T (W - W*)
        xT_E = x @ E                       # shape (d,)
        E_new = E - coef * jnp.outer(x, xT_E)
        return E_new, risk(E_new)

    keys = jax.random.split(stream_key, n_steps)
    E_final, risks = jax.lax.scan(step, E0, keys)
    return jnp.concatenate([risk(E0)[None], risks])  # prepend t=0 risk


def solve_ode(rho0, gamma_tilde, t_max, dt):
    """Euler's method for d/dt rho = (-2g + g^2) rho on [0, t_max].

    This is the rate that matches the literal SGD update: the exact expected
    one-step contraction of ||E||_F^2 is (1 - 2g/d + g^2 (d+2)/d^2), so in the
    rescaled time t = (step)/d the risk obeys d/dt rho = (-2g + g^2) rho.
    (Equivalently 2*(-g + g^2/2), i.e. twice the originally stated coefficient.)
    """
    rate = -2.0 * gamma_tilde + gamma_tilde**2
    n = int(round(t_max / dt))
    ts = np.linspace(0.0, t_max, n + 1)
    rho = np.empty(n + 1)
    rho[0] = rho0
    for k in range(n):
        rho[k + 1] = rho[k] + dt * rate * rho[k]
    return ts, rho


def empirical_for_d(d):
    """Fixed W*, W(0); N_RUNS streaming-SGD risk curves. Returns (curves, rho0)."""
    d_key = jax.random.fold_in(jax.random.PRNGKey(SEED), d)
    k_wstar, k_w0, k_streams = jax.random.split(d_key, 3)
    std = 1.0 / d                                      # var = 1/d^2
    W_star = std * jax.random.normal(k_wstar, (d, d))
    W0 = std * jax.random.normal(k_w0, (d, d))
    E0 = W0 - W_star                                   # same across all N_RUNS runs
    n_steps = T_FACTOR * d

    run_keys = jax.random.split(k_streams, N_RUNS)
    curves = []
    for run_key in run_keys:
        r = run_sgd(E0, run_key, n_steps, d, GAMMA_TILDE)
        curves.append(np.asarray(jax.block_until_ready(r)))
    return np.stack(curves), float(risk(E0))           # (N_RUNS, n_steps+1), rho0


def load_or_compute():
    """Empirical curves are the expensive part; cache them so theory/plot tweaks are free."""
    cache = dict(np.load(CACHE_NPZ)) if os.path.exists(CACHE_NPZ) else {}
    expected = {"config": np.array([N_RUNS, T_FACTOR, SEED, int(GAMMA_TILDE * 1e6)])}
    valid = ("config" in cache and np.array_equal(cache["config"], expected["config"])
             and all(f"curves_{d}" in cache for d in DS))
    if not valid:
        cache = dict(expected)
        for d in DS:
            print(f"  running SGD for d={d} ...", flush=True)
            curves, rho0 = empirical_for_d(d)
            cache[f"curves_{d}"] = curves
            cache[f"rho0_{d}"] = np.array(rho0)
        np.savez(CACHE_NPZ, **cache)
        print(f"cached empirical curves -> {CACHE_NPZ}")
    else:
        print(f"loaded cached empirical curves from {CACHE_NPZ}")
    return cache


def main():
    cache = load_or_compute()
    # one distinct color per d
    cmap = plt.cm.viridis(np.linspace(0.0, 0.85, len(DS)))
    fig, ax = plt.subplots(figsize=(9, 6))

    for color, d in zip(cmap, DS):
        curves = cache[f"curves_{d}"]                  # (N_RUNS, n_steps+1)
        n_steps = T_FACTOR * d
        mean = curves.mean(axis=0)
        std_curve = curves.std(axis=0)
        x = np.arange(n_steps + 1) / d                 # iterations / d, in [0, 20]

        ax.fill_between(x, mean - std_curve, mean + std_curve,
                        color=color, alpha=0.25, linewidth=0)
        ax.plot(x, mean, color=color, lw=1.8,
                label=f"empirical  d={d}")

        # --- theory: single ODE curve per d ---
        rho0 = float(cache[f"rho0_{d}"])
        ts, rho = solve_ode(rho0, GAMMA_TILDE, T_FACTOR, ODE_DT)
        ax.plot(ts, rho, color=color, lw=1.6, ls="--",
                label=f"theory (ODE)  d={d}")

    ax.set_yscale("log")
    ax.set_xlabel("SGD iterations / d")
    ax.set_ylabel(r"prediction risk  $R(W)=\frac{1}{2}\|W-W^\star\|_F^2$")
    ax.set_title(r"Streaming SGD risk vs. ODE theory ($\tilde\gamma=1$)")
    ax.grid(True, which="both", ls=":", alpha=0.4)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, ncol=2, fontsize=8, framealpha=0.9,
              loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)
    print(f"saved {OUT_PNG}")


if __name__ == "__main__":
    main()
