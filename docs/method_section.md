# Method: Factor-Constrained Identity Recomposition

## 1. Problem Formulation
Let $\mathcal{D}_A$ and $\mathcal{D}_B$ be two disjoint distributions of facial identities. Our goal is to generate a new identity $I_C$ such that:
1.  **Realism**: $I_C \in \mathcal{I}_{real}$ (The set of realistic face images).
2.  **Relatedness**: $I_C$ shares interpretable semantic factors with $\mathcal{D}_A$ and $\mathcal{D}_B$.
3.  **Anonymity**: For any recognition function $\phi(\cdot)$ and threshold $\tau$, $sim(\phi(I_C), \phi(I_A)) < \tau$ and $sim(\phi(I_C), \phi(I_B)) < \tau$, for all $I_A \in \mathcal{D}_A, I_B \in \mathcal{D}_B$.

## 2. Factor Decomposition
We define an identity as a composition of disjoint factors $F = \{f_{geom}, f_{sym}, f_{tex}, f_{local}\}$. A decomposition function $E(x) \rightarrow \{z_f\}_{f \in F}$ maps an image to a set of factor embeddings.
-   **Geometry ($z_{geom}$)**: Derived from sparse landmarks $L \in \mathbb{R}^{68 \times 2}$.
-   **Texture ($z_{tex}$)**: Derived from multi-scale Gabor filter responses on the canonically aligned face.
-   **Symmetry ($z_{sym}$)**: Derived from the residual $|I - flip(I)|$.

## 3. Recomposition
We introduce a Recomposition Module $R: \mathbb{R}^{|F| \times d} \rightarrow \mathcal{W}+$ that maps the set of factor embeddings to the extended latent space of a pretrained StyleGAN generator $G$.
$$ w = R(\{z_f\}) $$
$$ I_C = G(w) $$

To generate a mixed identity $I_C$, we sample factors from the source distributions. Let $\alpha_f \in [0, 1]$ be a mixing coefficient for factor $f$.
$$ z_f^C = \alpha_f z_f^A + (1-\alpha_f) z_f^B $$

## 4. Safety Constraints
We enforce anonymity via rejection sampling using an ensemble of recognition models $\Phi = \{\phi_1, \dots, \phi_K\}$. The Identity Leakage $\mathcal{L}_{leak}$ is defined as:
$$ \mathcal{L}_{leak}(I_C) = \max_{k, S \in \{A, B\}} \max_{I_S \in \mathcal{D}_S} \cos(\phi_k(I_C), \phi_k(I_S)) $$
A sample is accepted iff $\mathcal{L}_{leak}(I_C) < \tau$.

## 5. Identity Contribution Score (ICS)
To quantify the provenance of $I_C$, we define the Identity Contribution Score for factor $f$ towards source $S$:
$$ ICS_f(I_S \to I_C) = \frac{\exp(-d(z_f^C, z_f^S))}{\sum_{S' \in \{A,B\}} \exp(-d(z_f^C, z_f^{S'}))} $$
This metric provides a disentangled view of similarity, robust to global appearance changes.
