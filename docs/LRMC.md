# LRMC: Learnable-Radius Margin-Contrastive loss

LaTeX-ready methodology text, matching exactly what `src/lrmc/losses/lrmc_loss.py`
implements (see `docs/DECISIONS.md` for why this supersedes the earlier draft
formulation in `/LRMC_Zero-Day_Malware_Paper.md`).

## 1. Embedding and distance

Each input image $x_i$ is mapped through the ViT backbone $f$, projection head $h$,
and embedding normalizer to a point on the unit hypersphere:

```latex
z_i = \operatorname{normalize}\big(h(f(x_i))\big) \in \mathbb{S}^{d-1}.
```

The distance between an embedding and a class prototype $p$ is, by default, cosine
distance (equivalent to squared Euclidean distance divided by two, on the sphere):

```latex
d(z, p) = 1 - \langle z, p \rangle
\;=\; \tfrac{1}{2}\lVert z - p \rVert_2^2 \quad \text{(for } \lVert z\rVert=\lVert p\rVert=1\text{).}
```

A Euclidean-distance variant, $d(z,p) = \tfrac12\lVert z-p\rVert_2^2$ computed
directly (not requiring unit-norm inputs), is available as a config-selectable
ablation.

## 2. Prototypes

For each known class $c \in \{1,\dots,K\}$, the prototype $p_c$ is the (re-normalized)
mean of that class's embeddings, maintained as a **stop-gradient buffer**:

```latex
p_c \leftarrow \operatorname{normalize}\!\Big( m\, p_c + (1-m)\,
\operatorname{mean}_{i:\,y_i=c}\, z_i \Big), \qquad m = 0.9 \text{ (EMA mode, default)}
```

or, in batch-mean mode, $p_c \leftarrow \operatorname{normalize}(\operatorname{mean}_{i:y_i=c}\, z_i)$
with no memory across batches. A class absent from the current batch keeps its
previous value; a class's prototype is initialized from the first batch in which
it appears.

## 3. Radii

Each known class has a learnable radius parameterized to stay positive:

```latex
r_c = \operatorname{softplus}(\rho_c), \qquad \rho_c \in \mathbb{R} \text{ learnable},
```

initialized so that $r_c = r_{\text{init}}$ at $t=0$ (config default $0.5$). Radii
live in their own optimizer parameter group with a smaller learning rate and zero
weight decay, so they move slowly relative to the encoder.

## 4. Loss

```latex
\mathcal{L} = \mathcal{L}_{\text{supcon}} + \alpha\, \mathcal{L}_{\text{in}}
+ \beta\, \mathcal{L}_{\text{out}} + \gamma\, \mathcal{L}_{\text{rad}}
```

**Supervised contrastive term** (Khosla et al., 2020), two augmented views per
sample, temperature $\tau$:

```latex
\mathcal{L}_{\text{supcon}} = \sum_i \frac{-1}{|P(i)|} \sum_{p \in P(i)}
\log \frac{\exp(z_i \cdot z_p / \tau)}{\sum_{a \neq i} \exp(z_i \cdot z_a / \tau)}
```

where $P(i)$ is the set of same-family samples (across both views) in the batch,
excluding $i$ itself.

**Compactness term** -- pulls each embedding toward its own class's acceptance
sphere:

```latex
\mathcal{L}_{\text{in}} = \operatorname{mean}_i \big[\, d(z_i, p_{y_i}) - r_{y_i} \,\big]_+
```

**Separation term** -- pushes each embedding out of every *other* class's sphere,
normalized by the number of negative classes so it does not scale with $K$:

```latex
\mathcal{L}_{\text{out}} = \operatorname{mean}_i \left[
\frac{1}{K-1} \sum_{c \neq y_i} \big[\, r_c + \text{margin} - d(z_i, p_c) \,\big]_+
\right]
```

**Tightness term** -- a direct penalty on radius size, preventing the degenerate
solution where every sphere grows to swallow the whole embedding space:

```latex
\mathcal{L}_{\text{rad}} = \operatorname{mean}_c\, r_c
```

Gradients: $p_c$ is treated as a constant inside the loss (stop-gradient buffer,
updated separately by the Class Prototype Estimator); $z_i$ receives gradients
from all four terms and backpropagates into the projection head and backbone;
$r_c$ receives gradients from $\mathcal{L}_{\text{in}}$, $\mathcal{L}_{\text{out}}$
and $\mathcal{L}_{\text{rad}}$ and backpropagates only into $\rho_c$.

## 5. Radius-as-quantile: derivation

**Claim.** At a stationary point of $\alpha\mathcal{L}_{\text{in}} +
\gamma\mathcal{L}_{\text{rad}}$ with respect to $r_c$ (holding $z_i$ and $p_c$ fixed,
and under balanced class sizes $n_c \approx N/K$), $r_c$ equals approximately the
$(1 - \gamma/\alpha)$-quantile of the in-class distance distribution
$\{d(z_i, p_c) : y_i = c\}$.

**Argument.** Write $\mathcal{L}_{\text{in}} = \frac1N \sum_c \sum_{i: y_i=c}
[d_i - r_c]_+$ where $d_i := d(z_i, p_{y_i})$. The hinge $[d_i - r_c]_+$ has
subgradient $-\mathbb{1}[d_i > r_c]$ with respect to $r_c$ (it is $-1$ where the
point sits outside the sphere and is actively penalized, $0$ where it is already
inside). Summing over class $c$'s $n_c$ samples and writing $F_c$ for the empirical
CDF of $\{d_i : y_i = c\}$:

```latex
\frac{\partial}{\partial r_c}\Big[\alpha \mathcal{L}_{\text{in}}\Big]
= -\alpha \frac{n_c}{N} \big(1 - F_c(r_c)\big),
\qquad
\frac{\partial}{\partial r_c}\Big[\gamma \mathcal{L}_{\text{rad}}\Big] = \frac{\gamma}{K}.
```

Setting the sum to zero (stationarity):

```latex
\alpha \frac{n_c}{N} \big(1 - F_c(r_c)\big) = \frac{\gamma}{K}
\quad\Longrightarrow\quad
1 - F_c(r_c) = \frac{\gamma N}{\alpha K n_c}.
```

Under balanced classes, $n_c \approx N/K$, so the class-size factor cancels and

```latex
F_c(r_c) \approx 1 - \frac{\gamma}{\alpha},
```

i.e. $r_c$ sits at the $(1-\gamma/\alpha)$-quantile of the in-class distance
distribution, and $\gamma/\alpha$ is (approximately) the resulting **target
false-rejection rate** on the known class at training time -- the fraction of a
class's own training points whose distance exceeds its own radius. (The general,
class-imbalance-aware form is $1 - F_c(r_c) \approx \gamma N / (\alpha K n_c)$; an
under-represented class -- small $n_c$ -- ends up with a *higher* target FRR for
the same $\gamma,\alpha$, which is a real, reportable effect of class imbalance on
the radii and is one of the diagnostics to check tomorrow if radii look off.)

This is the same subgradient argument that gives $\nu$-SVM / SVDD's $\nu$
parameter its "fraction of margin violations" interpretation (Tax & Duin, 2004;
Ruff et al., 2018); LRMC's radius plays the analogous role, per class, with
$\gamma/\alpha$ standing in for $\nu$.

The separation term $\mathcal{L}_{\text{out}}$ also depends on $r_c$ (through
*other* classes' samples being pushed away from class $c$'s sphere), which adds a
second, $\beta$-weighted downward pressure on $r_c$ not captured in the argument
above; the derivation therefore describes the $\beta \to 0$ limit exactly and is
an approximation for $\beta > 0$. `tests/unit/test_lrmc_math.py::test_radius_quantile_equilibrium`
verifies the $\beta=0$ case numerically on synthetic embeddings, confirming the
claim holds (see that test for the measured tolerance).

## 6. Inference decision rule

At inference the network, prototypes and radii are frozen. For an embedding $z$:

```latex
s(x) = \min_c \frac{d(z, p_c)}{r_c}
```

The sample is **known**, predicted family $\hat y = \arg\min_c d(z,p_c)/r_c$, if
$s(x) \le \kappa$ (default $\kappa=1$, tuned on known-validation data only);
otherwise it is flagged **zero-day**. The alternative score $\min_c d(z,p_c)$
(ignoring radii) is also exposed, for AUROC comparisons against methods that have
no notion of a per-class radius. Post-hoc radius calibration (replacing the
learned $r_c$ with a per-class quantile of known-validation distances) is
implemented as a separate ablation in `eval/calibration.py`.
