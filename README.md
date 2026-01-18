# Explainable, Constrained, Identity-Safe Synthetic Identity Generation

## Mission
To implement a research prototype for generating "Person C" identities that are:
1.  **Photorealistic** (High fidelity)
2.  **Related** to source distributions A and B (Statistically mixed)
3.  **Identity-Safe** (Not identifiable as specific individuals from A or B)
4.  **Explainable** (Via factor decomposition and attribution maps)

## Abstract: The Digital Mask Maker

Imagine you want to create a new person for a movie or a game, but you don't want them to look exactly like any real actor.

**What we built:**
A smart computer program that takes a group of people (Group A) and another group (Group B) and creates a **brand new person ("Person C")**.

**How it works (The Recipe):**
1.  **Selection**: The computer picks the *eyes* from someone in Group A, the *face shape* from someone in Group B, and the *skin texture* from someone else.
2.  **Mixing**: It mixes these ingredients together like a master chef.
3.  **The Safety Check**: Before showing the new face, a strict "Security Guard" checks it against everyone in Group A and Group B.
    *   If the new face looks too much like a real person, the Guard says **"REJECTED!"** and the computer tries again.
    *   If the face is unique and new, the Guard says **"PASSED!"**

**why it matters:**
This allows researchers to share realistic data without revealing anyone's private identity. It's like wearing a perfect digital mask that looks real but isn't you.

**Technical Summary (for Researchers):**
"This is a **disentangled representation learning framework** utilizing **StyleGAN2-ADA**. We extract independent latent factors (Geometry, Texture, Symmetry) from two disjoint source distributions. We then recompose these factors to generate a synthetic identity $I_C$. Crucially, we enforce an **$\epsilon$-differential privacy-inspired constraint** using an ensemble of face recognition models (ArcFace, MagFace) to ensure the maximum cosine similarity between $I_C$ and any source sample remains below a strict threshold $\tau$, guaranteeing identity safety while preserving statistical realism."

## Safety & Ethics
-   **No Deepfakes**: This tool is NOT for impersonation. It uses constrained decomposition to create novel identities.
-   **Leakage Control**: We enforce a strict identity leakage threshold $\tau$. Any generated identity too close to a source is rejected.
-   **Watermarking**: All outputs should be treated as synthetic.

## Quick Start
1.  Install dependencies:
    ```bash
    python3 -m pip install -r requirements.txt
    ```
2.  Run the demo (generates safe identities from synthetic priors):
    ```bash
    python run_demo.py --num_samples 5 --tau 0.4
    ```

## Architecture
-   **Backbone**: StyleGAN2-ADA (Pretrained on FFHQ)
-   **Factor Decomposition**: Geometry, Symmetry, Texture, Local Features
-   **Identity Contribution Score (ICS)**: A principled metric for attributing "likeness" without identity theft.

## License
Research use only. Uses pretrained weights which may have their own non-commercial licenses (e.g., NVLabs StyleGAN).
