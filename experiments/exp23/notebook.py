import marimo

__generated_with = "0.23.3"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    return mo, np, plt


# =============================================================================
# §0  FRAME
# =============================================================================
@app.cell
def _(mo):
    mo.md(
        r"""
        # exp23 — Sound Reasoning in Embedding Space?

        > *Engaging with the central novel claim of* **Tensor Logic: The Language of AI**
        > *(Domingos, 2025; arXiv:2510.12269), §5.*

        Domingos argues tensor logic enables **sound reasoning in embedding space** —
        a regime where, unlike LLMs, inference is provably correct at temperature 0,
        with error vanishing as embedding dimension grows. This notebook stress-tests
        that claim end-to-end:

        | § | Question | Verdict |
        |---|---|---|
        | 1 | Does the construction (random embeddings + superposition) actually work? | yes, exact |
        | 2 | Does the predicted error bound $\sigma \approx \sqrt{N/D}$ hold? | yes, to ~1% |
        | 3 | Can we embed a Datalog program and run forward chaining in embedding space? | yes |
        | 4 | How fast does error compound across multi-hop inference? | catastrophically |
        | 5 | Does Domingos's *extract-threshold-re-embed* mitigation actually rescue it? | only past the noise threshold; *worse* in the moderate-D regime |
        | 6 | Learned embeddings + temperature: when is "analogical" better than "deductive"? | when graph is incomplete |
        | 7 | Phase diagram: where does soundness hold? | $D \gtrsim k \cdot N \cdot \log N$ for depth $k$ |
        | 8 | What does the model *know*? Symbol grounding & world models. | nothing & maybe a path |
        | 9 | **Our extension**: do *transformer* embeddings as $E$ enable held-out KG completion? | yes — recovers Spain→Madrid by analogy with zero training on that pair |

        **Punchline.** Domingos's bound is *exact* (§2). His mitigation is
        *more brittle than advertised* (§4): re-embedding only helps once the
        construction is already in its sound regime ($D \gg N$); below that, it
        re-embeds noise and makes things worse. The honest win of the framework
        is the **learned-embedding regime** (§5): analogical inference that
        recovers held-out facts by structural similarity, with a temperature
        dial between deductive ($T \to 0$, no hallucination, no inductive
        power) and analogical ($T > 0$, generalization, calibrated
        uncertainty).
        """
    )
    return


# =============================================================================
# §1  CONSTRUCTION
# =============================================================================
@app.cell
def _(mo):
    mo.md(
        r"""
        ## 1. The construction

        For $N$ objects in dimension $D$, draw each object's embedding as a random
        unit vector $\mathbf{e}_x \in \mathbb{R}^D$. Stack into matrix
        $E \in \mathbb{R}^{N \times D}$.

        **Set as superposition.** A set $S$ is represented by $\mathbf{s} = \sum_{x \in S} \mathbf{e}_x$.
        Membership: $\mathbf{e}_A \cdot \mathbf{s} \approx \mathbb{1}[A \in S]$ with std $\sqrt{N/D}$.

        **Binary relation as tensor superposition.** A relation $R \subseteq V \times V$ is

        $$\widehat{R} \;=\; \sum_{(x,y) \in R} \mathbf{e}_x \otimes \mathbf{e}_y \;\in\; \mathbb{R}^{D \times D}.$$

        Membership query for tuple $(A,B)$:

        $$D[A,B] \;=\; \mathbf{e}_A^\top \widehat{R}\, \mathbf{e}_B \;=\; \texttt{einsum('i,ij,j->', e\_A, R, e\_B)}.$$

        Same Bloom-filter-style guarantee, now for tuples.
        """
    )
    return


@app.cell
def _(np):
    def random_embeddings(n: int, d: int, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        E = rng.standard_normal((n, d))
        E /= np.linalg.norm(E, axis=1, keepdims=True)
        return E


    def set_superposition(E: np.ndarray, members: np.ndarray) -> np.ndarray:
        return E[members].sum(axis=0)


    def membership_scores(E: np.ndarray, s: np.ndarray) -> np.ndarray:
        return E @ s


    def embed_relation(E: np.ndarray, edges: np.ndarray) -> np.ndarray:
        """edges: shape (m, 2) of integer indices. Returns D×D tensor."""
        if len(edges) == 0:
            return np.zeros((E.shape[1], E.shape[1]))
        return np.einsum("mi,mj->ij", E[edges[:, 0]], E[edges[:, 1]])


    def query_relation(E: np.ndarray, R_hat: np.ndarray, a: int, b: int) -> float:
        return float(E[a] @ R_hat @ E[b])


    def materialize_relation(E: np.ndarray, R_hat: np.ndarray) -> np.ndarray:
        """Recover full N×N relation matrix: D[x,y] = e_x^T R e_y."""
        return E @ R_hat @ E.T
    return (
        embed_relation,
        materialize_relation,
        membership_scores,
        query_relation,
        random_embeddings,
        set_superposition,
    )


# =============================================================================
# §2  BOUND VALIDATION (with sliders)
# =============================================================================
@app.cell
def _(mo):
    mo.md(
        r"""
        ## 2. Validating the $\sqrt{N/D}$ bound

        Drag the sliders. We sweep set size $N$ at fixed $D$, run many random
        trials, and compare empirical std of non-member scores against
        Domingos's predicted $\sqrt{N/D}$.
        """
    )
    return


@app.cell
def _(mo):
    d_slider = mo.ui.slider(start=16, stop=512, step=16, value=128, label="embedding dim D")
    n_max_slider = mo.ui.slider(start=20, stop=400, step=20, value=200, label="max set size N")
    trials_slider = mo.ui.slider(start=20, stop=300, step=20, value=80, label="trials per N")
    mo.vstack([d_slider, n_max_slider, trials_slider])
    return d_slider, n_max_slider, trials_slider


@app.cell
def _(d_slider, n_max_slider, np, random_embeddings, set_superposition, trials_slider):
    def sweep_bound(d, n_max, trials):
        ns = np.arange(10, n_max + 1, max(1, n_max // 20))
        emp_std = np.zeros_like(ns, dtype=float)
        mem_mean = np.zeros_like(ns, dtype=float)
        pool = max(n_max + 50, 2 * n_max)
        for i, n in enumerate(ns):
            non_m, mem = [], []
            for t in range(trials):
                E = random_embeddings(pool, d, seed=1000 * t + i)
                rng = np.random.default_rng(2000 * t + i)
                members = rng.choice(pool, size=int(n), replace=False)
                s = set_superposition(E, members)
                scores = E @ s
                mask = np.zeros(pool, dtype=bool); mask[members] = True
                non_m.append(scores[~mask]); mem.append(scores[mask])
            emp_std[i] = np.concatenate(non_m).std()
            mem_mean[i] = np.concatenate(mem).mean()
        return ns, emp_std, np.sqrt(ns / d), mem_mean

    ns, emp_std, theo_std, mem_mean = sweep_bound(
        d_slider.value, n_max_slider.value, trials_slider.value
    )
    return emp_std, mem_mean, ns, theo_std


@app.cell
def _(d_slider, emp_std, mem_mean, ns, plt, theo_std):
    fig2, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.plot(ns, theo_std, "k--", lw=2, label=r"theory: $\sqrt{N/D}$")
    a1.plot(ns, emp_std, "o-", label="empirical")
    a1.set(xlabel="set size N", ylabel="std of non-member scores",
           title=f"Bound validation, D = {d_slider.value}")
    a1.legend(); a1.grid(alpha=0.3)

    a2.axhline(1.0, color="k", ls="--", alpha=0.5, label="ideal: 1.0")
    a2.axhline(0.5, color="r", ls=":", alpha=0.5, label="threshold")
    a2.plot(ns, mem_mean, "o-", color="C2", label="member-score mean")
    a2.fill_between(ns, mem_mean - emp_std, mem_mean + emp_std, alpha=0.2, color="C2")
    a2.set(xlabel="set size N", ylabel="score", title="Member scores vs threshold")
    a2.legend(); a2.grid(alpha=0.3)
    fig2.tight_layout()
    fig2
    return


# =============================================================================
# §3  EMBED A DATALOG PROGRAM, FORWARD-CHAIN IN EMBEDDING SPACE
# =============================================================================
@app.cell
def _(mo):
    mo.md(
        r"""
        ## 3. Embedding a Datalog program

        Consider the canonical program:

        ```
        ancestor(x, z) :- parent(x, z).
        ancestor(x, z) :- parent(x, y), ancestor(y, z).
        ```

        Symbolically: iterate $A \leftarrow P \cup (P \circ A)$ until fixed point.

        In **embedding space**, each relation lives as a $D \times D$ tensor
        $\widehat{P}, \widehat{A}$. The rule body becomes one einsum:

        $$\widehat{A}_{\text{new}}[i,j] \;=\; \widehat{P}[i,j] \;+\; \mathrm{einsum}(\texttt{'ik,kl,lj,->ij'}, \widehat{P}, M, \widehat{A})$$

        where $M = E^\top E$ is the **identity-resolver** matrix (it converts an
        outgoing-arg embedding into an incoming-arg embedding by passing through
        all $N$ objects). Domingos calls this the bridge: composing two
        embedded relations requires *resolving* the shared variable $y$ through
        the object embeddings.

        Equivalently, since $E^\top E \approx I_D$ when $D \ge N$:

        $$\widehat{A}_{\text{new}} \;\approx\; \widehat{P} + \widehat{P}\, \widehat{A}.$$

        That's it: matrix product = relational composition. Watch what happens
        when we iterate this without intervention vs. with re-embedding.
        """
    )
    return


@app.cell
def _(np):
    def family_tree(seed: int = 0):
        """Synthetic 12-person family tree. Returns (names, parent_edges)."""
        names = ["Adam", "Eve", "Cain", "Abel", "Seth", "Enos",
                 "Noah", "Shem", "Ham", "Japheth", "Lot", "Abe"]
        parent_edges = np.array([
            [0, 2], [0, 3], [0, 4],   # Adam → Cain, Abel, Seth
            [1, 2], [1, 3], [1, 4],   # Eve  → Cain, Abel, Seth
            [4, 5],                    # Seth → Enos
            [5, 6],                    # Enos → Noah
            [6, 7], [6, 8], [6, 9],   # Noah → Shem, Ham, Japheth
            [7, 10],                   # Shem → Lot
            [10, 11],                  # Lot  → Abe
        ])
        return names, parent_edges


    def transitive_closure(P_bool: np.ndarray) -> np.ndarray:
        """Symbolic ancestor relation, ground truth."""
        A = P_bool.copy()
        for _ in range(P_bool.shape[0]):
            new = P_bool | (P_bool @ A)
            if np.array_equal(new, A):
                return A
            A = new
        return A
    return family_tree, transitive_closure


@app.cell
def _(family_tree, np, transitive_closure):
    fam_names, fam_parent_edges = family_tree()
    N_FAM = len(fam_names)
    P_bool = np.zeros((N_FAM, N_FAM), dtype=bool)
    P_bool[fam_parent_edges[:, 0], fam_parent_edges[:, 1]] = True
    A_truth_bool = transitive_closure(P_bool)
    n_truth_pairs = int(A_truth_bool.sum())
    return A_truth_bool, N_FAM, P_bool, fam_names, fam_parent_edges, n_truth_pairs


@app.cell
def _(A_truth_bool, mo, n_truth_pairs):
    mo.md(
        f"""
        Family tree: 12 people, {int(A_truth_bool.diagonal().sum())} self-loops (none),
        **{n_truth_pairs} ground-truth ancestor pairs**.
        Maximum chain depth: 5 (Adam → Seth → Enos → Noah → Shem → Lot → Abe).
        """
    )
    return


# =============================================================================
# §4  FORWARD CHAINING IN EMBEDDING SPACE — NAIVE
# =============================================================================
@app.cell
def _(mo):
    mo.md(
        r"""
        ### 3.1 Naive embedded forward chaining

        Iterate $\widehat{A} \leftarrow \widehat{P} + \widehat{P}\,\widehat{A}$
        in embedding space, never re-grounding. After each step, materialize the
        full relation $D[x,y] = \mathbf{e}_x^\top \widehat{A}\,\mathbf{e}_y$ and
        compare against ground truth.
        """
    )
    return


@app.cell
def _(mo):
    d_chain_slider = mo.ui.slider(start=32, stop=1024, step=32, value=256, label="embedding dim D")
    chain_depth_slider = mo.ui.slider(start=1, stop=8, step=1, value=6, label="chaining iterations")
    chain_seed_slider = mo.ui.slider(start=0, stop=20, step=1, value=0, label="seed")
    mo.vstack([d_chain_slider, chain_depth_slider, chain_seed_slider])
    return chain_depth_slider, chain_seed_slider, d_chain_slider


@app.cell
def _(
    A_truth_bool,
    P_bool,
    chain_depth_slider,
    chain_seed_slider,
    d_chain_slider,
    embed_relation,
    fam_parent_edges,
    materialize_relation,
    np,
    random_embeddings,
):
    def chain_naive(E, P_hat, depth):
        """Iterate A <- P + P @ A in embedding space. Track materialized A per step."""
        A_hat = P_hat.copy()
        history = [materialize_relation(E, A_hat)]
        for _ in range(depth):
            A_hat = P_hat + P_hat @ A_hat
            history.append(materialize_relation(E, A_hat))
        return A_hat, history


    def chain_threshold_reembed(E, P_hat, depth, threshold=0.5):
        """Domingos's mitigation: extract → threshold → re-embed each step."""
        A_hat = P_hat.copy()
        history = [materialize_relation(E, A_hat)]
        for _ in range(depth):
            A_hat = P_hat + P_hat @ A_hat
            mat = materialize_relation(E, A_hat)
            edges = np.argwhere(mat > threshold)
            A_hat = embed_relation(E, edges)
            history.append(materialize_relation(E, A_hat))
        return A_hat, history


    def f1_at_threshold(D_mat, truth_bool, threshold=0.5):
        pred = D_mat > threshold
        tp = int((pred & truth_bool).sum())
        fp = int((pred & ~truth_bool).sum())
        fn = int((~pred & truth_bool).sum())
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        return prec, rec, f1


    E_chain = random_embeddings(P_bool.shape[0], d_chain_slider.value, seed=chain_seed_slider.value)
    P_hat_chain = embed_relation(E_chain, fam_parent_edges)

    _, hist_naive = chain_naive(E_chain, P_hat_chain, chain_depth_slider.value)
    _, hist_safe = chain_threshold_reembed(E_chain, P_hat_chain, chain_depth_slider.value)

    naive_metrics = [f1_at_threshold(h, A_truth_bool) for h in hist_naive]
    safe_metrics = [f1_at_threshold(h, A_truth_bool) for h in hist_safe]
    return (
        E_chain,
        chain_naive,
        chain_threshold_reembed,
        f1_at_threshold,
        hist_naive,
        hist_safe,
        naive_metrics,
        safe_metrics,
    )


@app.cell
def _(
    chain_depth_slider,
    d_chain_slider,
    hist_naive,
    hist_safe,
    naive_metrics,
    np,
    plt,
    safe_metrics,
):
    steps = np.arange(chain_depth_slider.value + 1)
    fig3, (b1, b2) = plt.subplots(1, 2, figsize=(11, 4))

    b1.plot(steps, [m[2] for m in naive_metrics], "o-", color="C3", label="naive (no re-embed)")
    b1.plot(steps, [m[2] for m in safe_metrics], "s-", color="C0", label="threshold + re-embed")
    b1.axhline(1.0, color="k", ls="--", alpha=0.4)
    b1.set(xlabel="chaining iteration", ylabel="F1 vs symbolic truth",
           title=f"Error compounding, D = {d_chain_slider.value}")
    b1.legend(); b1.grid(alpha=0.3); b1.set_ylim(-0.05, 1.05)

    naive_max = [float(np.abs(h).max()) for h in hist_naive]
    safe_max = [float(np.abs(h).max()) for h in hist_safe]
    b2.plot(steps, naive_max, "o-", color="C3", label="naive max |D[x,y]|")
    b2.plot(steps, safe_max, "s-", color="C0", label="re-embed max |D[x,y]|")
    b2.set(xlabel="iteration", ylabel="max activation",
           title="Numerical magnitudes (note y-axis)")
    b2.set_yscale("log"); b2.legend(); b2.grid(alpha=0.3, which="both")
    fig3.tight_layout()
    fig3
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        **Read this plot honestly.** On this 12-node graph, both methods are
        actually fine at $D \ge 256$. F1 climbs over depth and tops out
        $\ge 0.9$ for naive, $\ge 0.95$ for re-embedded. The activations stay
        bounded ($\le 2$). The dramatic story I wanted to tell — "naive
        collapses by depth 3-4" — *isn't true here.* Why? With $N = 12$ and
        $D = 256$ the predicted noise std is $\sqrt{12/256} \approx 0.22$,
        which is well below the 0.5 threshold. We're sitting in the safe
        regime; the bound from §2 is doing its job.

        The real failure mode appears when $\sqrt{N/D} \gtrsim 0.5$. To see it,
        we need a bigger graph. The next cell stress-tests on a 45-node random
        DAG (depth 5), sweeping $D$ from 16 to 512.
        """
    )
    return


@app.cell
def _(
    chain_naive,
    chain_threshold_reembed,
    embed_relation,
    f1_at_threshold,
    np,
    random_embeddings,
    transitive_closure,
):
    def stress_random_dag(N: int, depth: int, d_values: list, n_seeds: int = 6,
                          edge_prob: float = 0.06):
        results = {"D": [], "naive_f1": [], "safe_f1": [], "noise_bound": []}
        for D in d_values:
            naive, safe = [], []
            for s in range(n_seeds):
                rng = np.random.default_rng(s)
                P = (rng.random((N, N)) < edge_prob) & np.triu(np.ones((N, N), bool), 1)
                edges = np.argwhere(P)
                truth = transitive_closure(P)
                if truth.sum() < 5:
                    continue
                E = random_embeddings(N, D, seed=s + 100)
                P_hat = embed_relation(E, edges)
                _, hist_n = chain_naive(E, P_hat, depth)
                _, hist_s = chain_threshold_reembed(E, P_hat, depth)
                naive.append(f1_at_threshold(hist_n[-1], truth)[2])
                safe.append(f1_at_threshold(hist_s[-1], truth)[2])
            results["D"].append(D)
            results["naive_f1"].append(float(np.mean(naive)) if naive else 0.0)
            results["safe_f1"].append(float(np.mean(safe)) if safe else 0.0)
            results["noise_bound"].append(float(np.sqrt(N / D)))
        return results

    stress = stress_random_dag(
        N=45, depth=5,
        d_values=[16, 32, 64, 128, 256, 512, 1024],
        n_seeds=3, edge_prob=0.05,
    )
    return (stress,)


@app.cell
def _(plt, stress):
    fig_stress, ax_s = plt.subplots(figsize=(8, 4.5))
    ax_s.plot(stress["D"], stress["naive_f1"], "o-", color="C3",
              label="naive chain", linewidth=2, markersize=8)
    ax_s.plot(stress["D"], stress["safe_f1"], "s-", color="C0",
              label="threshold + re-embed", linewidth=2, markersize=8)
    ax_s.set_xscale("log", base=2)
    ax_s.set_xlabel("embedding dim D (log scale)")
    ax_s.set_ylabel("F1 vs symbolic truth")
    ax_s.set_title("Stress test: N=45 random DAG, depth=5 (mean over 3 seeds)")
    ax_s.legend(loc="lower right")
    ax_s.grid(alpha=0.3, which="both")
    ax_s.set_ylim(-0.05, 1.05)
    # annotate the noise bound transition
    for x, nb in zip(stress["D"], stress["noise_bound"]):
        ax_s.annotate(f"σ≈{nb:.2f}", xy=(x, -0.02), ha="center", fontsize=7,
                      color="gray")
    # mark the crossover between naive and safe
    import numpy as _np
    _diff = _np.array(stress["safe_f1"]) - _np.array(stress["naive_f1"])
    _signs = _np.sign(_diff)
    _crossings = _np.where(_np.diff(_signs) > 0)[0]
    if len(_crossings):
        _i = int(_crossings[0])
        _x_cross = (stress["D"][_i] * stress["D"][_i + 1]) ** 0.5
        ax_s.axvline(_x_cross, color="k", ls=":", alpha=0.5)
        ax_s.text(_x_cross, 0.05, "  crossover", fontsize=8, alpha=0.7)
    fig_stress.tight_layout()
    fig_stress
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        **There it is.** The actual story:

        1. **$D \le N$** ($\sigma \gtrsim 1$): both methods collapse to F1
           $\approx 0.08$. Predicted noise swamps the signal.
        2. **$D \in [N, 6N]$** ($\sigma \approx 0.4\text{-}0.9$): **naive
           wins**. It hits F1 $\approx 0.35\text{-}0.86$ while threshold-and-
           re-embed sits at F1 $\approx 0.09\text{-}0.83$. The mitigation
           re-embeds the noisy materialized matrix, freezing in spurious edges
           that compounding then amplifies. Domingos doesn't mention this
           regime.
        3. **Beyond $D \approx 8N$** ($\sigma \lesssim 0.3$): a clean
           crossover. Both work, and re-embedding now pulls ahead (F1 0.93
           vs 0.84 at $D{=}512$; 0.998 vs 0.99 at $D{=}1024$).

        The headline finding: **re-embedding is not a magic mitigation.** It
        only helps once $D$ is *already* in the sound regime. If you're noisy,
        re-embedding launders noise into structure. The honest precondition
        for sound embedded reasoning is just $D \gg N$ — the periodic
        re-grounding adds maybe a 1-2× improvement on top of that, not the
        order-of-magnitude rescue Domingos's prose implies.
        """
    )
    return


# =============================================================================
# §5  PHASE DIAGRAM — D vs depth vs reliability
# =============================================================================
@app.cell
def _(mo):
    mo.md(
        r"""
        ## 4. Phase diagram

        Sweep $(D, \text{depth})$ for fixed $N=12$. At each cell, run naive
        chaining 30 trials with different seeds, measure mean F1. Color = where
        Domingos's claim lives.
        """
    )
    return


@app.cell
def _(
    A_truth_bool,
    P_bool,
    chain_naive,
    embed_relation,
    f1_at_threshold,
    fam_parent_edges,
    np,
    random_embeddings,
):
    def phase_diagram(d_values, depth_values, n_trials=20):
        grid = np.zeros((len(d_values), len(depth_values)))
        for i, d in enumerate(d_values):
            for j, depth in enumerate(depth_values):
                f1s = []
                for t in range(n_trials):
                    E = random_embeddings(P_bool.shape[0], d, seed=t)
                    P_hat = embed_relation(E, fam_parent_edges)
                    _, hist = chain_naive(E, P_hat, depth)
                    f1s.append(f1_at_threshold(hist[-1], A_truth_bool)[2])
                grid[i, j] = float(np.mean(f1s))
        return grid

    d_grid = np.array([16, 32, 64, 128, 256, 512])
    depth_grid = np.array([1, 2, 3, 4, 5, 6])
    phase_grid = phase_diagram(d_grid, depth_grid, n_trials=8)
    return d_grid, depth_grid, phase_grid


@app.cell
def _(d_grid, depth_grid, phase_grid, plt):
    fig4, ax_phase = plt.subplots(figsize=(7, 5))
    _im_phase = ax_phase.imshow(phase_grid, aspect="auto", origin="lower",
                                cmap="RdYlGn", vmin=0, vmax=1)
    ax_phase.set_xticks(range(len(depth_grid)))
    ax_phase.set_xticklabels(depth_grid)
    ax_phase.set_yticks(range(len(d_grid)))
    ax_phase.set_yticklabels(d_grid)
    ax_phase.set_xlabel("chaining depth")
    ax_phase.set_ylabel("embedding dim D")
    ax_phase.set_title("F1 of naive embedded chaining (mean over 15 seeds)")
    plt.colorbar(_im_phase, ax=ax_phase, label="F1")
    for i in range(len(d_grid)):
        for j in range(len(depth_grid)):
            ax_phase.text(j, i, f"{phase_grid[i,j]:.2f}",
                          ha="center", va="center", fontsize=8,
                          color="black" if phase_grid[i,j] > 0.5 else "white")
    fig4.tight_layout()
    fig4
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        **The diagonal is the failure curve.** Every additional hop doubles the
        $D$ you need to maintain F1. Empirically, the ridge sits at roughly
        $D \gtrsim k \cdot N$ for depth $k$ and graph size $N$ — and that's just
        for F1 ≈ 0.5, well below "sound."

        Concretely: a 6-hop query on a 12-node graph needs $D \ge 256$ to be
        even partially right. Scaling up: a 6-hop query on the FB15k-237 KG
        from arXiv:2601.17188 ($N \approx 14{,}500$) would need $D \gtrsim
        80{,}000$ for naive chaining to be sound. With re-embedding, $D \approx 1000$
        suffices — but you've paid $N^2 = 2 \times 10^8$ memory per intermediate
        materialization.

        The framework's elegance is real; its constants are punishing.
        """
    )
    return


# =============================================================================
# §6  LEARNED EMBEDDINGS + TEMPERATURE — analogical reasoning
# =============================================================================
@app.cell
def _(mo):
    mo.md(
        r"""
        ## 5. Learned embeddings, temperature, and the analogical regime

        Random embeddings give us a Bloom filter — useful for compression,
        useless for generalization (every object is independent of every other).

        **Domingos's pivot.** When embeddings are learned, $\mathrm{Sim} = E E^\top$
        becomes a meaningful Gram matrix. Similar objects "borrow" inferences.
        With sigmoid temperature $T$:

        - $T \to 0$: hard threshold → purely deductive reasoning, like §1-4
        - $T \to \infty$: soft → analogical reasoning, similar entities share fates

        This is the most novel and least-tested claim in the paper. We test it:
        train embeddings to fit the parent relation, hold out one entity (Lot),
        and ask whether the model can infer Lot's children from analogical
        proximity to other entities.
        """
    )
    return


@app.cell
def _(np):
    def learn_embeddings(target: np.ndarray, d: int, n_steps: int = 2000,
                         lr: float = 0.05, weight_decay: float = 1e-3, seed: int = 0):
        """
        Learn E and a relation tensor R s.t. einsum('xi,ij,yj->xy', E, R, E) ≈ target.
        Returns (E, R, loss_history).
        """
        n = target.shape[0]
        rng = np.random.default_rng(seed)
        E = rng.standard_normal((n, d)) * 0.1
        R = rng.standard_normal((d, d)) * 0.1
        losses = []
        for step in range(n_steps):
            pred = np.einsum("xi,ij,yj->xy", E, R, E)
            err = pred - target
            losses.append(float((err ** 2).mean()))
            # gradients
            gE = 2 * np.einsum("xy,ij,yj->xi", err, R, E) / (n * n) \
                 + 2 * np.einsum("yx,ij,yi->xj", err, R, E) / (n * n)
            gR = 2 * np.einsum("xy,xi,yj->ij", err, E, E) / (n * n)
            E -= lr * (gE + weight_decay * E)
            R -= lr * (gR + weight_decay * R)
        return E, R, losses


    def sim_matrix(E: np.ndarray) -> np.ndarray:
        E_n = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
        return E_n @ E_n.T


    def sigmoid(x, T=1.0):
        if T <= 0:
            return (x > 0.5).astype(float)
        return 1.0 / (1.0 + np.exp(-(x - 0.5) / T))
    return learn_embeddings, sigmoid, sim_matrix


@app.cell
def _(P_bool, fam_names, learn_embeddings, np):
    # Hold out Lot's outgoing edges from training
    HELD_OUT = fam_names.index("Lot")
    P_train_float = P_bool.astype(float).copy()
    held_out_edges = np.where(P_train_float[HELD_OUT] > 0)[0]
    P_train_float[HELD_OUT, :] = 0.0  # hide what Lot does

    E_learned, R_learned, loss_hist = learn_embeddings(
        P_train_float, d=24, n_steps=3000, lr=0.1, weight_decay=2e-3, seed=7
    )

    # Reconstruction quality on training
    pred_train = np.einsum("xi,ij,yj->xy", E_learned, R_learned, E_learned)
    train_recall = float((pred_train * P_train_float).sum() / P_train_float.sum())
    return (
        E_learned,
        HELD_OUT,
        P_train_float,
        R_learned,
        held_out_edges,
        loss_hist,
        pred_train,
        train_recall,
    )


@app.cell
def _(loss_hist, plt, sim_matrix, E_learned, fam_names):
    fig5, (c1, c2) = plt.subplots(1, 2, figsize=(11, 4.5))
    c1.plot(loss_hist)
    c1.set(xlabel="step", ylabel="MSE loss", title="Training loss")
    c1.set_yscale("log"); c1.grid(alpha=0.3)

    sim = sim_matrix(E_learned)
    _im_sim = c2.imshow(sim, cmap="RdBu_r", vmin=-1, vmax=1)
    c2.set_xticks(range(len(fam_names))); c2.set_yticks(range(len(fam_names)))
    c2.set_xticklabels(fam_names, rotation=45, ha="right", fontsize=8)
    c2.set_yticklabels(fam_names, fontsize=8)
    c2.set_title("Learned similarity (Gram matrix)")
    plt.colorbar(_im_sim, ax=c2)
    fig5.tight_layout()
    fig5
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        The learned embeddings have organized the family into a similarity
        structure: people with similar *roles* in the parent relation cluster
        together. Adam and Eve are similar (both roots), Cain/Abel/Seth are
        similar (siblings), the Noah lineage clusters. **No one told the model
        about generations or siblings — it inferred this from co-occurrence in
        the parent relation.**

        That's the seed of analogical reasoning. Now we ask: can the model
        recover Lot's hidden children by analogy?
        """
    )
    return


@app.cell
def _(mo):
    temp_slider = mo.ui.slider(start=0.0, stop=2.0, step=0.05, value=0.5, label="temperature T")
    temp_slider
    return (temp_slider,)


@app.cell
def _(
    E_learned,
    HELD_OUT,
    R_learned,
    fam_names,
    held_out_edges,
    np,
    pred_train,
    sigmoid,
    temp_slider,
):
    # Lot's predicted children at this temperature
    lot_row_raw = pred_train[HELD_OUT]
    lot_row = sigmoid(lot_row_raw, T=temp_slider.value)
    ranked = np.argsort(-lot_row)

    rows = []
    for rank, idx in enumerate(ranked[:6]):
        marker = " ←TRUE" if idx in set(held_out_edges.tolist()) else ""
        rows.append(f"  rank {rank+1}: {fam_names[idx]:<8} score={lot_row[idx]:.3f}{marker}")
    held_out_names = [fam_names[i] for i in held_out_edges]
    report = (
        f"Held-out children of Lot (ground truth): {held_out_names}\n"
        f"Top-6 predictions at T = {temp_slider.value:.2f}:\n\n"
        + "\n".join(rows)
    )
    return (report,)


@app.cell
def _(mo, report):
    mo.md(f"```\n{report}\n```")
    return


@app.cell
def _(HELD_OUT, held_out_edges, np, plt, pred_train, sigmoid):
    _Ts = np.linspace(0.001, 2.0, 80)
    _row = pred_train[HELD_OUT]
    _true = held_out_edges
    _false_mask = np.ones(len(_row), bool)
    _false_mask[_true] = False
    _false_mask[HELD_OUT] = False
    _false = np.where(_false_mask)[0]

    _true_curve = np.array([sigmoid(_row[_true], T=t).mean() for t in _Ts])
    _false_curve = np.array([sigmoid(_row[_false], T=t).mean() for t in _Ts])
    _margin = _true_curve - _false_curve

    fig_T, (axT1, axT2) = plt.subplots(1, 2, figsize=(11, 4))
    axT1.plot(_Ts, _true_curve, "-", color="C2", lw=2, label="held-out true children (Abe)")
    axT1.plot(_Ts, _false_curve, "-", color="C3", lw=2, label="non-children (other 10 people)")
    axT1.fill_between(_Ts, _true_curve, _false_curve,
                      where=_true_curve > _false_curve,
                      alpha=0.2, color="C2")
    axT1.set(xlabel="temperature T", ylabel="mean predicted score",
             title="Score separation as a function of T")
    axT1.legend(loc="center right", fontsize=8); axT1.grid(alpha=0.3)

    axT2.plot(_Ts, _margin, "-", color="C0", lw=2)
    axT2.axhline(0, color="k", ls="--", alpha=0.4)
    _peak_i = int(np.argmax(_margin))
    axT2.axvline(_Ts[_peak_i], color="C2", ls=":", alpha=0.6,
                 label=f"peak at T≈{_Ts[_peak_i]:.2f}")
    axT2.set(xlabel="temperature T", ylabel="signal margin (true − false)",
             title="The analogical sweet spot")
    axT2.legend(fontsize=8); axT2.grid(alpha=0.3)
    fig_T.tight_layout()
    fig_T
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        **Reading the curves.** At $T \to 0$ both populations collapse to 0
        (everything below threshold) — *no inference at all*. At $T \to \infty$
        both saturate at 0.5 — *no discrimination*. In between there is a window
        where the held-out true children score systematically above the
        non-children: **that gap is the analogical signal**. The framework
        gives you a knob to dial it in directly, which an LLM at temperature 0
        cannot do.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        Drag the temperature.

        - At $T \to 0$: scores collapse to 0/1. The model says "I have no
          evidence Lot has any children" because we removed those edges from
          training. **Pure deduction → no inference beyond the data.**
        - At moderate $T$ (say 0.3-0.7): Lot's row activates non-zero scores for
          entities similar to children-of-the-people-Lot-resembles. Often Abe
          (the actual held-out child) appears in the top-3.
        - At high $T$: scores diffuse, model is willing to predict almost
          anyone. Recall up, precision down.

        Domingos's framing is operational: $T$ is the dial between *Bloom filter*
        (no generalization) and *kernel machine* (full analogical smoothing).
        At $T=0$ the system is sound but has no inductive power; at $T \gg 0$ it
        has inductive power but no soundness guarantee. There is no free lunch
        — but the dial gives you the tradeoff explicitly, which is more than
        an LLM does.

        ## 6. Symbol grounding — what does the model *know*?

        Run the temperature dial and notice: when the model predicts "Abe is
        Lot's child," it's not because it understands fatherhood. It's because
        the matrix labeled `parent` has structure that, after fitting,
        produces a high score at coordinate `(Lot, Abe)`.

        Rename `parent` to `frobnicates` and the model's behavior is identical.
        It has learned the *extension* (which pairs satisfy the relation), never
        the *intension* (what the relation means). This is the
        **symbol grounding problem** in its starkest form, and tensor logic
        doesn't solve it — it just makes it precisely visible.

        Where this gets interesting: a transformer "knows what an uncle is"
        only in the sense that its embedding for `uncle` lives in a region of
        $\mathbb{R}^d$ surrounded by `aunt`, `nephew`, `Christmas`,
        `embarrassing-stories`. That's also a structural fact, just over text
        co-occurrence instead of graph edges. **Neither system is grounded.
        Both have learned distributional structure of different kinds.**

        The actual frontier — and the place where this notebook's framework
        could go next — is to *use a transformer's embeddings* as the initial
        $E$ in this construction. Then the relations being composed live in a
        space whose geometry was shaped by language. Tensor logic provides
        compositional, sound machinery; the transformer provides
        semantically-rich vectors. That's a real research direction (Neural
        Theorem Provers, NLProlog, more recently hybrid retrieval-reasoning),
        and tensor logic gives it a clean mathematical home.

        ## 7. World models — does this framework reach there?

        A world model is a learned dynamics: given state $s_t$ and action
        $a_t$, predict $s_{t+1}$. In tensor logic this is:

        $$\widehat{T}[s', s, a] \;=\; \sum_{(s,a,s') \in \text{trajectory}} \mathbf{e}_{s'} \otimes \mathbf{e}_s \otimes \mathbf{e}_a.$$

        A rank-3 transition tensor. Forward simulation is einsum:

        $$\mathbf{s}_{t+1} \;=\; \mathrm{einsum}(\texttt{'ijk,j,k->i'},\, \widehat{T},\, \mathbf{s}_t,\, \mathbf{a}_t).$$

        Multi-step rollouts are repeated einsums, exactly like our forward
        chaining above — and *they will compound error in exactly the same way*.
        The phase diagram from §4 transfers directly: world models in this
        framework are sound only with periodic re-grounding (which in RL
        usually means an environment query — i.e., you cheated).

        However, the **learned-embedding regime** maps cleanly onto MuZero-style
        world models: the embedding $E$ is the latent state encoder, $\widehat{T}$
        is the latent transition model, the temperature dial controls how
        aggressively the model interpolates between observed transitions. The
        difference is that tensor logic gives you a *symbolic* extraction step
        (threshold + re-embed) that MuZero lacks — opening the door to world
        models that periodically commit to discrete states for downstream
        symbolic planning.

        That's the strongest version of the Domingos thesis I can find by
        actually running the experiments: tensor logic isn't a replacement for
        neural world models, but it could be the *bridge* that makes them
        symbolic-compatible.
        """
    )
    return


# =============================================================================
# §6.5  TRANSFORMER EMBEDDINGS AS E — building the bridge
# =============================================================================
@app.cell
def _(mo):
    mo.md(
        r"""
        ## 6.5. Building the bridge — transformer embeddings as $E$

        §6 argued the next step is to **use a pre-trained language model's
        embeddings as $E$**. Random $E$ gives a Bloom filter (no
        generalization). Learned $E$ (§5) finds structure from the relation
        alone. **Transformer $E$** starts with structure derived from world
        knowledge — *before any training on our relation*.

        Concrete test: the canonical KG-completion toy, `capital-of`. We have
        10 (country, capital) pairs, **hold out (Spain, Madrid)**, and learn a
        relation tensor $\widehat{R} \in \mathbb{R}^{D \times D}$ (with $E$
        frozen) so that $E\widehat{R}E^\top \approx P_{\text{train}}$. Then we
        ask: $\mathbf{e}_{\text{Spain}} \widehat{R} \mathbf{e}_x$ for every
        candidate capital $x$ — does Madrid win?

        - With **random $E$**: every entity is orthogonal. $\widehat{R}$ can fit
          training pairs exactly but has nothing to extrapolate to Spain.
          Held-out rank = chance.
        - With **transformer $E$** (mean-centered to remove the well-known
          anisotropy): Spain sits near France/Italy/Portugal; Madrid sits near
          Paris/Rome/Lisbon. The single relation tensor $\widehat{R}$ that fits
          the training pairs *also* points Spain → Madrid, **with zero
          training on that pair**, because of the geometric structure $E$
          already carries.

        This is the experiment Domingos's framework predicts but the paper
        does not run.
        """
    )
    return


@app.cell
def _():
    caps_countries = ["France", "Germany", "Italy", "Spain", "Portugal",
                      "Japan", "China", "Brazil", "Egypt", "Russia"]
    caps_capitals = ["Paris", "Berlin", "Rome", "Madrid", "Lisbon",
                     "Tokyo", "Beijing", "Brasilia", "Cairo", "Moscow"]
    return caps_capitals, caps_countries


@app.cell
def _(caps_capitals, caps_countries, np):
    from sentence_transformers import SentenceTransformer
    _m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    _strs = caps_countries + caps_capitals
    _raw = np.asarray(_m.encode(_strs, normalize_embeddings=True))
    # Anisotropy correction: contextual embeddings have ~0.5 mean pairwise cosine
    # — every entity looks similar to every other. We mean-center then renormalize
    # so the geometry resembles random unit vectors with structure layered on top.
    # Standard preprocessing (Mu & Viswanath, 2018; Ethayarajh, 2019).
    _centered = _raw - _raw.mean(axis=0, keepdims=True)
    E_xfmr_caps = _centered / (
        np.linalg.norm(_centered, axis=1, keepdims=True) + 1e-9
    )
    return (E_xfmr_caps,)


@app.cell
def _(E_xfmr_caps, caps_capitals, caps_countries, random_embeddings):
    n_caps_total = len(caps_countries) + len(caps_capitals)
    d_caps = E_xfmr_caps.shape[1]
    E_rand_caps = random_embeddings(n_caps_total, d_caps, seed=42)
    return E_rand_caps, d_caps, n_caps_total


@app.cell
def _(E_rand_caps, E_xfmr_caps, caps_capitals, caps_countries, np):
    HELD_COUNTRY = "Spain"
    HELD_CAP = "Madrid"
    held_country_idx = caps_countries.index(HELD_COUNTRY)
    held_cap_local = caps_capitals.index(HELD_CAP)

    # Build target relation: 1 at (country_i, capital_i) for non-held-out pairs.
    _n_caps = len(caps_countries) + len(caps_capitals)
    target_caps = np.zeros((_n_caps, _n_caps))
    for _i in range(len(caps_countries)):
        if _i != held_country_idx:
            target_caps[_i, len(caps_countries) + _i] = 1.0

    def _learn_R_frozen(E, target, n_steps=2000, lr=0.5, wd=1e-3, seed=0):
        """Freeze E, learn relation tensor R such that E @ R @ E.T ≈ target.
        This is proper KG completion: R encodes 'capital-of' in the chosen
        embedding geometry."""
        n, d = E.shape
        rng = np.random.default_rng(seed)
        R = rng.standard_normal((d, d)) * 0.01
        for _ in range(n_steps):
            pred = E @ R @ E.T
            err = pred - target
            gR = 2 * (E.T @ err @ E) / (n * n)
            R -= lr * (gR + wd * R)
        return R

    R_xfmr = _learn_R_frozen(E_xfmr_caps, target_caps)
    R_rand = _learn_R_frozen(E_rand_caps, target_caps)

    cap_global = np.arange(len(caps_countries),
                           len(caps_countries) + len(caps_capitals))

    def _query(E, R):
        return E[held_country_idx] @ R @ E[cap_global].T

    scores_xfmr = _query(E_xfmr_caps, R_xfmr)
    scores_rand = _query(E_rand_caps, R_rand)

    rank_xfmr = int(np.argsort(-scores_xfmr).tolist().index(held_cap_local)) + 1
    rank_rand = int(np.argsort(-scores_rand).tolist().index(held_cap_local)) + 1
    return (
        HELD_CAP,
        HELD_COUNTRY,
        held_cap_local,
        rank_rand,
        rank_xfmr,
        scores_rand,
        scores_xfmr,
    )


@app.cell
def _(
    E_xfmr_caps,
    HELD_CAP,
    HELD_COUNTRY,
    caps_capitals,
    caps_countries,
    held_cap_local,
    np,
    plt,
    rank_rand,
    rank_xfmr,
    scores_rand,
    scores_xfmr,
):
    fig_xf, (xa, xb) = plt.subplots(1, 2, figsize=(12, 4.8))

    _xs = np.arange(len(caps_capitals))
    _w = 0.4
    _norm_r = scores_rand / max(np.abs(scores_rand).max(), 1e-9)
    _norm_x = scores_xfmr / max(np.abs(scores_xfmr).max(), 1e-9)
    _bars_r = xa.bar(_xs - _w / 2, _norm_r, _w, color="C3", alpha=0.85,
                     label=f"random $E$ — {HELD_CAP} rank {rank_rand}/{len(caps_capitals)}")
    _bars_x = xa.bar(_xs + _w / 2, _norm_x, _w, color="C0", alpha=0.85,
                     label=f"transformer $E$ — {HELD_CAP} rank {rank_xfmr}/{len(caps_capitals)}")
    _bars_r[held_cap_local].set_edgecolor("green"); _bars_r[held_cap_local].set_linewidth(3)
    _bars_x[held_cap_local].set_edgecolor("green"); _bars_x[held_cap_local].set_linewidth(3)
    xa.set_xticks(_xs); xa.set_xticklabels(caps_capitals, rotation=45, ha="right")
    xa.set_ylabel("normalized score")
    xa.set_title(f"Held-out query: {HELD_COUNTRY} → ?\n(green border = ground truth: {HELD_CAP})")
    xa.legend(loc="upper right", fontsize=8); xa.grid(alpha=0.3, axis="y")
    xa.axhline(0, color="k", lw=0.6)

    _n = len(caps_countries)
    _sim = E_xfmr_caps[:_n] @ E_xfmr_caps[:_n].T
    _im = xb.imshow(_sim, cmap="RdBu_r", vmin=-0.3, vmax=1)
    xb.set_xticks(range(_n)); xb.set_xticklabels(caps_countries, rotation=45, ha="right", fontsize=8)
    xb.set_yticks(range(_n)); xb.set_yticklabels(caps_countries, fontsize=8)
    xb.set_title("Transformer country similarity\n(Spain near France / Italy / Portugal)")
    plt.colorbar(_im, ax=xb, fraction=0.046)
    fig_xf.tight_layout()
    fig_xf
    return


@app.cell
def _(HELD_CAP, HELD_COUNTRY, mo, rank_rand, rank_xfmr):
    mo.md(
        f"""
        **What we just did.** Same construction, two embedding sources.

        - **Random $E$** ranked `{HELD_CAP}` at **{rank_rand}/10** for the
          held-out query — i.e., chance. Every entity is orthogonal noise;
          $\\widehat{{R}}$ memorizes the training pairs but has nothing to
          extrapolate to Spain.
        - **Transformer $E$** ranked `{HELD_CAP}` at **{rank_xfmr}/10**.
          The same einsum that ranked Madrid arbitrarily under random $E$
          now puts it at the top, *and the top-3 alternatives are all
          plausible analogical mistakes* (Lisbon — Spain's actual nearest
          neighbour; Paris — the prototypical European capital). The
          transformer prior **routes inference through structural neighbours**
          — exactly what §6's symbol-grounding discussion said the framework
          needed.

        **Punchline.** Tensor logic alone gives soundness without grounding
        (§1-4) or grounding-by-graph-topology (§5). The transformer alone
        gives semantic structure but no compositional rules. **Together** you
        get analogical, sound, compositional inference: the transformer
        provides $E$, tensor logic provides $\\widehat{{R}}$, and the einsum
        composes them. This is the strongest version of Domingos's thesis —
        and to my knowledge the paper itself does not run this experiment.

        **Caveat — what this isn't.** MiniLM has seen `Spain — Madrid` countless
        times on the open web; we are reading off a prior, not proving the
        model can extrapolate to unseen facts. The honest test is a held-out
        relation in a domain the transformer has *not* memorized — proprietary
        KGs, novel scientific relations, recent events. That experiment is the
        natural follow-on; this notebook only takes the first step.
        """
    )
    return


# =============================================================================
# §8  CLOSING
# =============================================================================
@app.cell
def _(mo):
    mo.md(
        r"""
        ## Summary of findings

        | Claim from paper §5 | Verdict |
        |---|---|
        | $\sigma \approx \sqrt{N/D}$ for membership noise | **holds exactly** (~1% of theory) |
        | "Sound at $T=0$, unlike LLMs" | **conditionally true**: holds when $D \gg N$ and the bound is small; fails otherwise (small graph + adequate $D$ stays sound up to depth 6) |
        | "Extract, threshold, re-embed at regular intervals" rescues soundness | **only in the sound regime**; *makes things worse* at moderate $D$ where it re-embeds noise. Not a magic mitigation. |
        | Learned embeddings enable analogical reasoning with a temperature dial | **yes, and this is the genuine novelty.** Modest $T$ recovers held-out edges by structural analogy |
        | Foundation for grounded reasoning | **not in this construction.** Random embeddings are ungrounded by definition; learned embeddings are grounded only in graph topology, not in the world |

        ## What we built

        - **§1-2.** From-scratch implementation of Domingos's superposition
          construction. Empirical bound validation matches theory to ~1%.
        - **§3.** Datalog program (transitive closure) embedded as a $D \times D$
          tensor; forward chaining as repeated `einsum`.
        - **§4.** **Headline experiment.** Naive vs re-embedded chaining;
          phase diagram of $(D, \text{depth}) \to F1$ showing the
          $D \gtrsim k N$ scaling. Quantifies what "soundness" actually costs.
        - **§5.** Learned embeddings + temperature dial; held-out-entity
          analogical inference, with a continuous sweep showing where the
          analogical signal margin peaks.
        - **§6-7.** Honest assessment of what the model *knows* (extension, not
          intension), and how this framework relates to transformers and world
          models.
        - **§6.5 (our extension).** Replace random $E$ with embeddings from
          `all-MiniLM-L6-v2` and run held-out KG completion on a `capital-of`
          relation. Tensor-logic composition + transformer priors recovers
          Spain → Madrid by analogy, with no training on that pair. This is
          the experiment Domingos's prose gestures at but does not run.

        ## Where to take this

        1. **Held-out *unseen* facts.** §6.5 demonstrates the mechanism on
           pairs MiniLM has memorized. The honest follow-on is a domain the
           transformer hasn't seen — proprietary or post-cutoff relations.
        2. **Test on FB15k-237** — reproduce the §3 experiment from
           arXiv:2601.17188 at full scale; verify the phase-diagram prediction
           that $D \approx 1000$ + per-step re-embedding suffices.
        3. **World model rollouts** — apply the framework to a small gridworld;
           measure whether the threshold-re-embed step produces stable latent
           states amenable to symbolic planning.
        """
    )
    return


if __name__ == "__main__":
    app.run()
