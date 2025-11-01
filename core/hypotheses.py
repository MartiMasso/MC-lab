
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from scipy.stats import (
    binomtest,
    t as tdist,
    chi2,
    norm,
    binom,
    poisson,
    beta,
)

# -----------------------------
# Helper utilities
# -----------------------------
def _fmt_float(x, nd=6):
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)


def _two_sided_binom_region(n, p0, alpha):
    """Return integer critical region (k <= k_inf or k >= k_sup) for a two-sided binomial test at level alpha using exact tails.
    We choose the smallest symmetric (by probability) rejection region with total tail probability <= alpha.
    """
    # Compute two-sided exact p-values by tail accumulation
    # Strategy: grow symmetric from the center until total tail <= alpha just before crossing.
    pmf = binom.pmf(np.arange(n + 1), n, p0)
    cdf = binom.cdf(np.arange(n + 1), n, p0)

    # Start from center around np0
    center = int(np.floor(n * p0))
    k_inf, k_sup = None, None
    # Try all possible symmetric widths and choose the region with total tail <= alpha and maximal acceptance
    best = None
    for a in range(center + 1):
        low = center - a
        high = n - (center - a)
        # acceptance = [low, high]; rejection = k <= low-1 or k >= high+1
        rej_left = cdf[low - 1] if low - 1 >= 0 else 0.0
        rej_right = 1 - cdf[high]
        total_rej = rej_left + rej_right
        if total_rej <= alpha:
            best = (low, high, total_rej)
            break
    if best is None:
        # Fallback to normal approx if exact loop failed for pathological inputs
        z = norm.ppf(1 - alpha / 2)
        sd = np.sqrt(n * p0 * (1 - p0))
        low_c = int(np.floor(n * p0 - z * sd - 0.5))
        high_c = int(np.ceil(n * p0 + z * sd + 0.5))
        return max(0, low_c), min(n, high_c)
    return best[0], best[1]


# -----------------------------
# Main app
# -----------------------------

def run_hypothesis_tests():
    st.subheader("📊 Pruebas de hipótesis — laboratorio interactivo")

    with st.expander("ℹ️ Qué hace este módulo / How to use"):
        st.markdown(
            r"""
**Objetivo.** Explorar y **contrastar hipótesis** con herramientas clásicas (frecuentistas) y también opciones **Bayesianas**, calcular **potencias**, **tamaños muestrales**, **regiones críticas** y límites para procesos de **Poisson** (p. ej., límites de vida media).

**Incluye:**
- **Binomial** (exacto): contraste, región crítica, potencia y cálculo de **n**.
- **t de Student (2 muestras, var. iguales)** y **Welch (var. desiguales)**.
- **χ²** de **bondad de ajuste** e **independencia** (tabla de contingencia).
- **Z-test de una media** (σ conocida) y **t-test de una media** (σ desconocida).
- **Poisson** (conteos): test de media, **límite superior** a C.L. dado y conversión a **límite inferior de vida media**.
- **Bayes Beta–Binomial**: posterior, prob. de sesgo (p>p₀) y visualización.

> Ajusta el **nivel α** y otras opciones en cada prueba y observa **p‑valores**, **potencias** y gráficos.
            """,
            unsafe_allow_html=True,
        )

    test_type = st.selectbox(
        "Selecciona el test / herramienta",
        [
            "Binomial (contraste & potencia)",
            "Tamaño muestral binomial (potencia)",
            "Z-test (1 muestra, σ conocida)",
            "t de Student (1 muestra)",
            "t de Student (2 muestras, var iguales)",
            "t de Welch (2 muestras, var desiguales)",
            "Chi-cuadrado (bondad de ajuste)",
            "Chi-cuadrado (independencia)",
            "Poisson (test & upper limit)",
            "Bayes Beta–Binomial",
        ],
    )

    # ---------------------------------
    # Binomial main tool
    # ---------------------------------
    if test_type == "Binomial (contraste & potencia)":
        st.markdown("### Binomial: contrast, critical region, power")
        c1, c2, c3 = st.columns(3)
        with c1:
            n = st.number_input("n (trials)", 1, 1_000_000, 250)
        with c2:
            k = st.number_input("k (successes)", 0, int(n), 140)
        with c3:
            p0 = st.number_input("p₀ (null)", 0.0, 1.0, 0.5)

        alt = st.selectbox("Alternative", ["two-sided", "greater", "less"], index=0)
        alpha = st.number_input("α (significance)", 0.000001, 0.5, 0.05, step=0.00001, format="%.6f")

        # Exact p-value via scipy
        result = binomtest(k, n, p0, alternative=alt)
        pval = result.pvalue
        st.write(f"**p-value:** {_fmt_float(pval)}")
        if pval < alpha:
            st.error(f"❌ Reject H₀ at α={alpha}")
        else:
            st.success(f"✅ Do not reject H₀ at α={alpha}")

        # Critical region (two-sided exact)
        if alt == "two-sided":
            k_inf, k_sup = _two_sided_binom_region(n, p0, alpha)
            st.info(
                f"Two-sided critical region (exact):  X ≤ **{k_inf}** or X ≥ **{k_sup}**."
            )
        elif alt == "greater":
            # Upper tail critical point (smallest k with tail ≤ alpha)
            k_sup = int(binom.isf(alpha, n, p0))  # inverse survival
            st.info(f"One-sided (≥): reject if X ≥ **{k_sup}**.")
            k_inf = None
        else:  # less
            k_inf = int(binom.ppf(alpha, n, p0))
            st.info(f"One-sided (≤): reject if X ≤ **{k_inf}**.")
            k_sup = None

        st.markdown("#### Power at an alternative p")
        p1 = st.number_input("p (alternative)", 0.0, 1.0, 0.55)
        # Compute power
        if alt == "two-sided":
            power = binom.cdf(k_inf, n, p1) + (1 - binom.cdf(k_sup - 1, n, p1))
        elif alt == "greater":
            power = 1 - binom.cdf(k_sup - 1, n, p1)
        else:
            power = binom.cdf(k_inf, n, p1)
        st.write(f"**Power at p={p1}:** {_fmt_float(power)}")

        # Plot pmf under H0 and H1 with critical region
        st.markdown("#### Plot settings")
        colp1, colp2 = st.columns(2)
        with colp1:
            plot_h = st.slider("Plot height (inches)", min_value=2.0, max_value=6.0, value=4.0, step=0.1)
        with colp2:
            y_max = st.number_input("Y-axis max (0 = auto)", min_value=0.0, value=0.0, step=0.00001, format="%.6f")

        # Auto-crop option: truncate x-axis where H0 pmf falls below a threshold
        crop = st.checkbox("Auto-crop x-axis by PMF threshold (H₀)", value=True)
        thresh = st.number_input("PMF threshold for cropping", min_value=0.0, value=0.00001, step=0.000001, format="%.6f")

        xs_all = np.arange(n + 1)
        pmf0 = binom.pmf(xs_all, n, p0)
        pmf1 = binom.pmf(xs_all, n, p1)

        fig, ax = plt.subplots(figsize=(8, plot_h))
        ax.bar(xs_all, pmf0, alpha=0.4, label=f"H₀: Bin(n,{p0})")
        ax.plot(xs_all, pmf1, lw=2, label=f"H₁: Bin(n,{p1})")
        if k_inf is not None:
            ax.axvspan(0, k_inf, color="C3", alpha=0.2, label="Reject left")
        if k_sup is not None:
            ax.axvspan(k_sup, n, color="C3", alpha=0.2, label="Reject right")
        ax.axvline(k, color="black", ls="--", label=f"Observed k={k}")
        ax.set_xlabel("k")
        ax.set_ylabel("Probability")
        if y_max > 0:
            ax.set_ylim(0, y_max)

        if crop:
            # Find contiguous region where pmf0 >= threshold
            idx = np.where(pmf0 >= thresh)[0]
            if idx.size > 0:
                pad = 3  # a few bins of padding on each side
                x_min = max(0, int(idx[0] - pad))
                x_max = min(n, int(idx[-1] + pad))
                ax.set_xlim(x_min, x_max)
        ax.legend(loc="best")
        st.pyplot(fig)

    # ---------------------------------
    # Binomial sample size for desired power
    # ---------------------------------
    elif test_type == "Tamaño muestral binomial (potencia)":
        st.markdown("### Sample size for binomial test (two-sided, α fixed)")
        p0 = st.number_input("p₀ (null)", 0.0, 1.0, 0.5)
        p1 = st.number_input("p (alternative)", 0.0, 1.0, 0.55)
        alpha = st.number_input("α (significance)", 0.000001, 0.5, 0.05, step=0.00001, format="%.6f")
        target_power = st.number_input("Target power", 0.5, 0.999, 0.9, step=0.01)

        # Brute-force n search with exact tails (keeps it simple and reliable for classroom sizes)
        max_n = st.number_input("Max n to search", 100, 100000, 5000, step=100)

        best_n, best_power = None, None
        for n in range(10, int(max_n) + 1):
            k_inf, k_sup = _two_sided_binom_region(n, p0, alpha)
            power = binom.cdf(k_inf, n, p1) + (1 - binom.cdf(k_sup - 1, n, p1))
            if power >= target_power:
                best_n, best_power = n, power
                break
        if best_n is not None:
            st.success(f"Minimum n achieving power ≥ {target_power}: **{best_n}** (power≈{_fmt_float(best_power)})")
        else:
            st.warning("No n found within the search range. Increase 'Max n' or relax targets.")

    # ---------------------------------
    # Z-test 1-sample (sigma known)
    # ---------------------------------
    elif test_type == "Z-test (1 muestra, σ conocida)":
        st.markdown("### One-sample Z-test (σ known)")
        mu0 = st.number_input("μ₀ (null mean)", value=0.0)
        xbar = st.number_input("x̄ (sample mean)", value=10.0)
        sigma = st.number_input("σ (known)", value=10.0)
        n = st.number_input("n", 1, 1_000_000, 25)
        alt = st.selectbox("Alternative", ["two-sided", "greater", "less"])
        alpha = st.number_input("α", 0.000001, 0.5, 0.05, step=0.00001, format="%.6f")

        se = sigma / np.sqrt(n)
        z = (xbar - mu0) / se
        if alt == "two-sided":
            pval = 2 * (1 - norm.cdf(abs(z)))
            zcrit = norm.ppf(1 - alpha / 2)
            reject = abs(z) > zcrit
        elif alt == "greater":
            pval = 1 - norm.cdf(z)
            zcrit = norm.ppf(1 - alpha)
            reject = z > zcrit
        else:
            pval = norm.cdf(z)
            zcrit = norm.ppf(alpha)
            reject = z < zcrit
        st.write(f"**z = {z:.3f}**, **p-value = {_fmt_float(pval)}**")
        st.write(f"Critical z: **{_fmt_float(zcrit)}** → {'Reject' if reject else 'Do not reject'} H₀")

    # ---------------------------------
    # t-tests
    # ---------------------------------
    elif test_type == "t de Student (1 muestra)":
        st.markdown("### One-sample t-test (σ unknown)")
        n = st.number_input("n", 2, 1_000_000, 15)
        xbar = st.number_input("x̄ (sample mean)", value=10.0)
        s = st.number_input("s (sample std)", value=2.0)
        mu0 = st.number_input("μ₀ (null)", value=9.0)
        alt = st.selectbox("Alternative", ["two-sided", "greater", "less"])
        alpha = st.number_input("α", 0.000001, 0.5, 0.05, step=0.00001, format="%.6f")

        se = s / np.sqrt(n)
        t_stat = (xbar - mu0) / se
        df = n - 1
        if alt == "two-sided":
            pval = 2 * (1 - tdist.cdf(abs(t_stat), df))
            tcrit = tdist.ppf(1 - alpha / 2, df)
            reject = abs(t_stat) > tcrit
        elif alt == "greater":
            pval = 1 - tdist.cdf(t_stat, df)
            tcrit = tdist.ppf(1 - alpha, df)
            reject = t_stat > tcrit
        else:
            pval = tdist.cdf(t_stat, df)
            tcrit = tdist.ppf(alpha, df)
            reject = t_stat < tcrit
        st.write(f"**t = {t_stat:.3f}**, **df = {df}**, **p-value = {_fmt_float(pval)}**")
        st.write(f"Critical t: **{_fmt_float(tcrit)}** → {'Reject' if reject else 'Do not reject'} H₀")

    elif test_type == "t de Student (2 muestras, var iguales)":
        st.markdown("### Two-sample t-test (pooled variances)")
        n1 = st.number_input("n₁", 2, 1_000_000, 15)
        mean1 = st.number_input("Mean 1", value=10.0)
        s1 = st.number_input("SD 1", value=2.0)
        n2 = st.number_input("n₂", 2, 1_000_000, 12)
        mean2 = st.number_input("Mean 2", value=9.0)
        s2 = st.number_input("SD 2", value=2.5)
        alt = st.selectbox("Alternative", ["two-sided", "greater", "less"])
        alpha = st.number_input("α", 0.000001, 0.5, 0.05, step=0.00001, format="%.6f")

        sp2 = ((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2)
        t_stat = (mean1 - mean2) / np.sqrt(sp2 * (1 / n1 + 1 / n2))
        df = n1 + n2 - 2

        if alt == "two-sided":
            pval = 2 * (1 - tdist.cdf(abs(t_stat), df))
            tcrit = tdist.ppf(1 - alpha / 2, df)
            reject = abs(t_stat) > tcrit
        elif alt == "greater":
            pval = 1 - tdist.cdf(t_stat, df)
            tcrit = tdist.ppf(1 - alpha, df)
            reject = t_stat > tcrit
        else:
            pval = tdist.cdf(t_stat, df)
            tcrit = tdist.ppf(alpha, df)
            reject = t_stat < tcrit

        st.write(f"**t = {t_stat:.3f}**, **df = {df}**, **p-value = {_fmt_float(pval)}**")
        st.write(f"Critical t: **{_fmt_float(tcrit)}** → {'Reject' if reject else 'Do not reject'} H₀")

    elif test_type == "t de Welch (2 muestras, var desiguales)":
        st.markdown("### Welch t-test (unequal variances)")
        n1 = st.number_input("n₁", 2, 1_000_000, 15)
        mean1 = st.number_input("Mean 1", value=10.0)
        s1 = st.number_input("SD 1", value=2.0)
        n2 = st.number_input("n₂", 2, 1_000_000, 12)
        mean2 = st.number_input("Mean 2", value=9.0)
        s2 = st.number_input("SD 2", value=2.5)
        alt = st.selectbox("Alternative", ["two-sided", "greater", "less"])
        alpha = st.number_input("α", 0.000001, 0.5, 0.05, step=0.00001, format="%.6f")

        se = np.sqrt((s1 ** 2) / n1 + (s2 ** 2) / n2)
        t_stat = (mean1 - mean2) / se
        # Welch–Satterthwaite df
        df = (se ** 2) ** 2 / (
            ((s1 ** 2) / n1) ** 2 / (n1 - 1) + ((s2 ** 2) / n2) ** 2 / (n2 - 1)
        )

        if alt == "two-sided":
            pval = 2 * (1 - tdist.cdf(abs(t_stat), df))
            tcrit = tdist.ppf(1 - alpha / 2, df)
            reject = abs(t_stat) > tcrit
        elif alt == "greater":
            pval = 1 - tdist.cdf(t_stat, df)
            tcrit = tdist.ppf(1 - alpha, df)
            reject = t_stat > tcrit
        else:
            pval = tdist.cdf(t_stat, df)
            tcrit = tdist.ppf(alpha, df)
            reject = t_stat < tcrit

        st.write(f"**t = {t_stat:.3f}**, **df ≈ {df:.1f}**, **p-value = {_fmt_float(pval)}**")
        st.write(f"Critical t: **{_fmt_float(tcrit)}** → {'Reject' if reject else 'Do not reject'} H₀")

    # ---------------------------------
    # Chi-square tests
    # ---------------------------------
    elif test_type == "Chi-cuadrado (bondad de ajuste)":
        st.markdown("### χ² goodness-of-fit")
        obs = st.text_input("Observed counts (comma-separated)", "50, 30, 20")
        probs = st.text_input("Expected probabilities (comma-separated)", "0.5, 0.3, 0.2")
        obs = np.array([float(x) for x in obs.split(",")])
        probs = np.array([float(x) for x in probs.split(",")])
        probs = probs / np.sum(probs)
        exp = np.sum(obs) * probs

        chi2_stat = np.sum((obs - exp) ** 2 / exp)
        df = len(obs) - 1
        pval = 1 - chi2.cdf(chi2_stat, df)

        st.write(f"**χ² = {chi2_stat:.3f}**, **df = {df}**, **p-value = {_fmt_float(pval)}")
        if pval < 0.05:
            st.error("❌ Reject H₀ at 5%")
        else:
            st.success("✅ Do not reject H₀")

        fig, ax = plt.subplots()
        indices = np.arange(len(obs))
        ax.bar(indices - 0.15, obs, width=0.3, label="Observed")
        ax.bar(indices + 0.15, exp, width=0.3, label="Expected")
        ax.set_xticks(indices)
        ax.set_xticklabels([f"C{i+1}" for i in range(len(obs))])
        ax.set_ylabel("Count")
        ax.legend()
        st.pyplot(fig)

    elif test_type == "Chi-cuadrado (independencia)":
        st.markdown("### χ² test for independence (contingency table)")
        raw = st.text_area(
            "Enter the contingency table rows (comma-separated per row)",
            "10, 20\n15, 25",
        )
        rows = [r.strip() for r in raw.splitlines() if r.strip()]
        table = np.array([[float(x) for x in r.split(",")] for r in rows])
        rsum = table.sum(axis=1, keepdims=True)
        csum = table.sum(axis=0, keepdims=True)
        total = table.sum()
        expected = rsum @ csum / total
        chi2_stat = np.sum((table - expected) ** 2 / expected)
        df = (table.shape[0] - 1) * (table.shape[1] - 1)
        pval = 1 - chi2.cdf(chi2_stat, df)
        st.write(f"**χ² = {chi2_stat:.3f}**, **df = {df}**, **p-value = {_fmt_float(pval)}")

        fig, ax = plt.subplots()
        im = ax.imshow((table - expected) / np.sqrt(expected + 1e-9), aspect="auto")
        ax.set_title("Standardized residuals")
        plt.colorbar(im, ax=ax)
        st.pyplot(fig)

    # ---------------------------------
    # Poisson counting experiments: tests and upper limits
    # ---------------------------------
    elif test_type == "Poisson (test & upper limit)":
        st.markdown("### Poisson: hypothesis test and upper limit (CL)")
        tabs = st.tabs(["Test of mean", "Upper limit μ (CL)", "Lifetime lower limit"])

        with tabs[0]:
            lam0 = st.number_input("λ₀ (null mean)", 0.0, 1e9, 5.0)
            x = st.number_input("Observed counts", 0, 10**9, 0)
            alt = st.selectbox("Alternative", ["two-sided", "greater", "less"], key="pois_alt")
            # Exact p-values
            if alt == "two-sided":
                # central two-sided by doubling the smaller one-sided tail
                p_lo = poisson.cdf(x, lam0)
                p_hi = 1 - poisson.cdf(x - 1, lam0) if x > 0 else 1.0
                pval = 2 * min(p_lo, p_hi)
                pval = min(1.0, pval)
            elif alt == "greater":
                pval = 1 - poisson.cdf(x - 1, lam0) if x > 0 else 1.0
            else:
                pval = poisson.cdf(x, lam0)
            st.write(f"**p-value:** {_fmt_float(pval)}")

        with tabs[1]:
            cl = st.number_input("Confidence level (e.g. 0.90)", 0.5, 0.9999, 0.90, step=0.01)
            x = st.number_input("Observed counts (background≈0)", 0, 10**9, 0, key="pois_ul_x")
            # Upper limit on μ given x obs: solve P(X<=x | μ_UL)=CL ⇒ μ_UL = qpois(x; CL)
            # Using inverse CDF: for given x, find μ s.t. CDF(x; μ)=CL  ⇒ μ = poisson.isf(1-CL, x)
            # scipy's isf works on k given μ; we invert via search
            # For small x, closed form: x=0 ⇒ μ_UL = -ln(1-CL)
            if x == 0:
                mu_ul = -np.log(1 - cl)
            else:
                # numeric search for μ such that CDF(x; μ)=cl
                # simple bracket and bisect
                lo, hi = 0.0, 1.0
                while poisson.cdf(x, hi) < cl:
                    hi *= 2.0
                    if hi > 1e12:
                        break
                for _ in range(80):
                    mid = 0.5 * (lo + hi)
                    if poisson.cdf(x, mid) >= cl:
                        hi = mid
                    else:
                        lo = mid
                mu_ul = 0.5 * (lo + hi)
            st.write(f"**μ_UL at CL={cl}:** {_fmt_float(mu_ul, 6)}")

        with tabs[2]:
            st.markdown("Convert exposure to a **lifetime lower limit**: μ = N·T/τ ⇒ τ > N·T / μ_UL")
            N = st.number_input("Number of targets N", 0.0, 1e40, 1.67e32)
            T_years = st.number_input("Live time T (years)", 0.0, 1e9, 2.738)
            mu_ul = st.number_input("μ_UL from previous tab", 0.0, 1e6, 2.302585)
            tau_ll = (N * T_years) / mu_ul if mu_ul > 0 else np.inf
            st.success(f"Lifetime lower limit: **τ > {_fmt_float(tau_ll, 3)} years**")

    # ---------------------------------
    # Bayesian Beta–Binomial
    # ---------------------------------
    else:  # Bayes Beta–Binomial
        st.markdown("### Bayesian Beta–Binomial")
        n = st.number_input("n (trials)", 1, 10_000_000, 250)
        k = st.number_input("k (successes)", 0, int(n), 140)
        a0 = st.number_input("Prior α (Beta)", 0.001, 10_000.0, 1.0)
        b0 = st.number_input("Prior β (Beta)", 0.001, 10_000.0, 1.0)
        p0 = st.number_input("Threshold p₀ for P(p>p₀)", 0.0, 1.0, 0.5)

        a_post = a0 + k
        b_post = b0 + (n - k)
        prob_gt = 1 - beta.cdf(p0, a_post, b_post)

        st.write(
            f"Posterior: Beta(α={_fmt_float(a_post,3)}, β={_fmt_float(b_post,3)}) — "
            f"**P(p>p₀) = {_fmt_float(prob_gt, 4)}**"
        )

        # Plot prior and posterior
        xs = np.linspace(0, 1, 400)
        prior_pdf = beta.pdf(xs, a0, b0)
        post_pdf = beta.pdf(xs, a_post, b_post)
        fig, ax = plt.subplots()
        ax.plot(xs, prior_pdf, label=f"Prior Beta({a0},{b0})", lw=2)
        ax.plot(xs, post_pdf, label=f"Posterior Beta({int(a_post)},{int(b_post)})", lw=2)
        ax.axvline(p0, color="k", ls="--", label="p₀")
        ax.fill_between(xs, 0, post_pdf, where=xs >= p0, alpha=0.2, label="P(p>p₀)")
        ax.set_xlabel("p")
        ax.set_ylabel("density")
        ax.legend()
        st.pyplot(fig)
