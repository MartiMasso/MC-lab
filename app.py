import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from scipy.stats import norm

from core.rng import mlcg_sample
from core.gaussian import sum12_gaussian, box_muller
from core.inverse import sample_exponential
from core.rejection import sample_rejection
from core.mcmc import metropolis, acf, ising_metropolis
from core.integration import mc_crude, mc_stratified, mc_importance
from core.weights import weighted_histogram, equivalent_events
from core.rambo import rambo_massless
from core.hypotheses import run_hypothesis_tests

st.set_page_config(page_title="Monte Carlo Lab", layout="wide")
st.title("Monte Carlo Lab — v1")

st.sidebar.header("Controles generales")
seed = st.sidebar.number_input("Seed", min_value=0, value=12345, step=1)
np.random.seed(seed)

tabs = st.tabs([
    "Uniformes & MLCG",
    "Gaussianas",
    "1D (Inversa & Rechazo)",
    "MCMC (Metropolis)",
    "Integración MC",
    "Eventos ponderados",
    "RAMBO",
    "Ising 2D",
    "Hipótesis"
])

# ------------------- TAB 1: Uniformes & MLCG -------------------
with tabs[0]:
    st.subheader("Uniformes: NumPy vs MLCG")
    n = st.number_input("n (muestras)", min_value=100, max_value=500000, value=5000, step=500)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**NumPy RNG**")
        u_np = np.random.random(n)
        fig, ax = plt.subplots()
        ax.hist(u_np, bins=50, density=True)
        ax.set_title("Uniformes NumPy")
        st.pyplot(fig)

    with col2:
        st.markdown("**MLCG (congruencial lineal)**")
        u_ml = mlcg_sample(n, seed=seed)
        fig, ax = plt.subplots()
        ax.hist(u_ml, bins=50, density=True)
        ax.set_title("Uniformes MLCG")
        st.pyplot(fig)

    st.markdown("**Mapa (u_i, u_{i+1})** para MLCG (visualiza patrones finos)")
    m = min(n, 5000)
    fig, ax = plt.subplots()
    ax.scatter(u_ml[:m-1], u_ml[1:m], s=4, alpha=0.4)
    ax.set_xlabel("u_i")
    ax.set_ylabel("u_{i+1}")
    st.pyplot(fig)

# ------------------- TAB 2: Gaussianas -------------------
with tabs[1]:
    st.subheader("Generación Gaussiana")
    n_g = st.number_input("n (muestras gaussianas)", min_value=100, max_value=200000, value=10000, step=500)
    method = st.selectbox("Método", ["Suma de 12 uniformes", "Box–Muller"])

    if method == "Suma de 12 uniformes":
        z = sum12_gaussian(n_g, rng=np.random)
    else:
        z = box_muller(n_g, rng=np.random)

    fig, ax = plt.subplots()
    ax.hist(z, bins=60, density=True, alpha=0.6, label="Muestras")
    xs = np.linspace(-4, 4, 400)
    ax.plot(xs, norm.pdf(xs), lw=2, label="N(0,1)")
    ax.set_title("Histograma vs pdf teórica")
    ax.legend()
    st.pyplot(fig)

# ------------------- TAB 3: 1D (Inversa & Rechazo) -------------------
with tabs[2]:
    st.subheader("1D: Transformada inversa y Rechazo")
    sub = st.radio("Selecciona", ["Exponencial (inversa)", "Rechazo (función libre)"], horizontal=True)

    if sub == "Exponencial (inversa)":
        tau = st.number_input("τ (tau)", min_value=0.001, value=1.0, step=0.1)
        n_e = st.number_input("n (muestras exp)", min_value=100, max_value=200000, value=10000, step=500)
        x = sample_exponential(tau, n_e, rng=np.random)

        fig, ax = plt.subplots()
        ax.hist(x, bins=60, density=True, alpha=0.6, label="Muestras")
        xs = np.linspace(0, np.percentile(x, 99.5), 400)
        ax.plot(xs, (1.0/tau)*np.exp(-xs/tau), lw=2, label="pdf teórica")
        ax.set_title("Exponencial por transformada inversa")
        ax.legend()
        st.pyplot(fig)

    else:
        st.markdown("**f(x) personalizada** en [a,b] con c ≥ max f(x). Ejemplo por defecto: f(x)=0.75(1-x²) en x∈[-1,1].")
        a = st.number_input("a", value=-1.0, step=0.1)
        b = st.number_input("b", value=1.0, step=0.1)
        c = st.number_input("c (cota superior de f)", value=0.75, step=0.05)
        n_r = st.number_input("n (muestras objetivo)", min_value=100, max_value=100000, value=5000, step=500)

        expr = st.text_input("f(x) (usa 'x' y funciones numpy, p.ej. 0.75*(1-x**2))", "0.75*(1-x**2)")

        def fwrap(xs: np.ndarray) -> np.ndarray:
            x = xs  # disponible para eval
            return np.array(eval(expr, {"np": np, "x": x}), dtype=float)

        xsamp, acc = sample_rejection(fwrap, a, b, c, n_r, rng=np.random)
        st.write(f"Tasa de aceptación: **{100*acc:.2f}%**")

        fig, ax = plt.subplots()
        ax.hist(xsamp, bins=50, density=True, alpha=0.6, label="Muestras aceptadas")
        grid = np.linspace(a, b, 400)
        y = fwrap(grid)
        ax.plot(grid, y / np.trapz(y, grid), lw=2, label="f(x) (normalizada aprox.)")
        ax.set_title("Rechazo: histograma vs forma de f(x)")
        ax.legend()
        st.pyplot(fig)

# ------------------- TAB 4: MCMC (Metropolis) -------------------
with tabs[3]:
    st.subheader("MCMC — Metropolis 1D (mezcla gaussiana)")
    st.markdown("Target: mezcla **p(x) = a·N(μ1,σ1) + (1-a)·N(μ2,σ2)**")

    a = st.slider("a (peso del primer pico)", 0.0, 1.0, 0.5, 0.05)
    mu1 = st.number_input("μ1", value=-2.0, step=0.1)
    s1 = st.number_input("σ1", value=0.7, step=0.05, min_value=0.05)
    mu2 = st.number_input("μ2", value=2.0, step=0.1)
    s2 = st.number_input("σ2", value=0.7, step=0.05, min_value=0.05)

    prop_std = st.slider("Std de la propuesta", 0.05, 3.0, 0.8, 0.05)
    n_chain = st.number_input("Longitud cadena", min_value=1000, max_value=200000, value=20000, step=1000)
    burn = st.number_input("Burn-in", min_value=0, max_value=100000, value=2000, step=500)
    x0 = st.number_input("x0 (inicio)", value=0.0, step=0.5)

    def logp(x: float) -> float:
        return np.log(a * norm.pdf(x, mu1, s1) + (1.0 - a) * norm.pdf(x, mu2, s2) + 1e-300)

    chain, acc = metropolis(logp, x0=x0, prop_std=prop_std, n=n_chain, rng=np.random)
    st.write(f"Tasa de aceptación: **{100*acc:.1f}%**")

    # Trazas
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots()
        ax.plot(chain[:min(2000, len(chain))], lw=1)
        ax.set_title("Traza (primeros pasos)")
        st.pyplot(fig)

    # Histograma vs densidad teórica
    with col2:
        fig, ax = plt.subplots()
        ax.hist(chain[burn:], bins=60, density=True, alpha=0.6, label="Cadena (post burn-in)")
        xs = np.linspace(min(chain)-4, max(chain)+4, 400)
        target = a * norm.pdf(xs, mu1, s1) + (1.0 - a) * norm.pdf(xs, mu2, s2)
        ax.plot(xs, target, lw=2, label="Target")
        ax.set_title("Distribución muestreada vs target")
        ax.legend()
        st.pyplot(fig)

# ------------------- TAB 5: MC Integration -------------------
with tabs[4]:
    st.subheader("Integración Monte Carlo")
    choice = st.selectbox("Esquema", ["Crudo", "Estratificado", "Importance"])
    st.markdown("g(x) por defecto: `np.sqrt(1 - x**2)` en [0,1] (→ π/4).")
    expr = st.text_input("g(x)", "np.sqrt(1.0 - x**2)")
    def gfun(x): 
        return eval(expr, {"np": np, "x": x})
    if choice == "Crudo":
        n = st.number_input("n", 1000, 2_000_000, 50_000, step=5000)
        I, err = mc_crude(gfun, 0.0, 1.0, n)
        st.write(f"**I ≈ {I:.6f} ± {err:.6f}**  (1σ)")
    elif choice == "Estratificado":
        k = st.slider("Estratos k", 2, 200, 10)
        n = st.number_input("n total", 1000, 2_000_000, 50_000, step=5000)
        I, err = mc_stratified(gfun, 0.0, 1.0, k, n)
        st.write(f"**I ≈ {I:.6f} ± {err:.6f}**  (1σ)")
    else:
        st.markdown("Ejemplo de importance con h(x)=Beta(2,2) en [0,1].")
        from scipy.stats import beta as beta_dist
        n = st.number_input("n", 1000, 2_000_000, 50_000, step=5000)
        def sampler_h(n): return np.random.beta(2, 2, size=n)
        def pdf_h(x): return beta_dist.pdf(x, 2, 2)
        I, err = mc_importance(gfun, sampler_h, pdf_h, n)
        st.write(f"**I ≈ {I:.6f} ± {err:.6f}**  (1σ)")

# ------------------- TAB 6: Weights -------------------
with tabs[5]:
    st.subheader("Eventos ponderados y N equivalente")
    st.markdown("Genera x~N(0,1) y define pesos w=f(x) (editables).")
    n = st.number_input("n", 1000, 500000, 50000, step=5000)
    x = np.random.normal(size=n)
    w_expr = st.text_input("w(x)", "np.exp(-0.5*x**2)")  # ejemplo
    w = eval(w_expr, {"np": np, "x": x})
    H, err, centers = weighted_histogram(x, w, bins=60, range=(-4,4))
    Neq = equivalent_events(w)
    st.write(f"**N_equivalente ≈ {Neq:.1f}**  (N real = {n})")
    fig, ax = plt.subplots()
    ax.bar(centers, H, width=(centers[1]-centers[0]), alpha=0.6, label="Histo ponderado")
    ax.errorbar(centers, H, yerr=err, fmt='.', lw=1, label="Error 1σ")
    ax.legend(); ax.set_title("Histograma ponderado con errores")
    st.pyplot(fig)

# ------------------- TAB 7: RAMBO -------------------
with tabs[6]:
    st.subheader("RAMBO — Massless phase space")
    N = st.slider("Número de partículas", 2, 12, 6)
    W = st.number_input("Energía total W", 1.0, 1000.0, 100.0)
    P4 = rambo_massless(N, W)
    st.write("Primero(s) 4-vectores (E, px, py, pz):")
    st.dataframe(P4[:min(N,10)], use_container_width=True)
    st.write("Chequeo de conservación (∑pᵢ):")
    st.write(np.sum(P4, axis=0))
    fig, ax = plt.subplots()
    costh = P4[:,3] / (np.linalg.norm(P4[:,1:], axis=1) + 1e-12)
    ax.hist(costh, bins=30, density=True, alpha=0.7)
    ax.set_title("Distribución cos(θ) (debería ser ~ uniforme)")
    st.pyplot(fig)

# ------------------- TAB 8: Ising 2D -------------------
with tabs[7]:
    st.subheader("Ising 2D — Metropolis")
    L = st.slider("L", 8, 64, 24, step=4)
    beta = st.number_input("β = 1/T", 0.05, 1.0, 0.45, step=0.05)
    sweeps = st.number_input("n_sweeps", 10, 5000, 500, step=50)
    S, E_hist, M_hist = ising_metropolis(L, beta, sweeps)
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots()
        im = ax.imshow(S, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_title("Spins finales")
        st.pyplot(fig)
    with col2:
        fig, ax = plt.subplots()
        ax.plot(E_hist, label="E por sitio"); ax.plot(M_hist, label="M por sitio")
        ax.legend(); ax.set_title("E y M vs sweeps")
        st.pyplot(fig)
    # Autocorrelación de M
    max_lag = min(100, len(M_hist)-1)
    rho = acf(M_hist - np.mean(M_hist), max_lag=max_lag)
    fig, ax = plt.subplots()
    ax.plot(np.arange(len(rho)), rho)
    ax.set_title("ACF(M), diagnóstico de correlación")
    st.pyplot(fig)


# ------------------- TAB 9: Pruebas de hipótesis -------------------
with tabs[8]:
    st.subheader("Pruebas de hipótesis")
    # Llama al módulo de hipótesis para renderizar su UI
    run_hypothesis_tests()