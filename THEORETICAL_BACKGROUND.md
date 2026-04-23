# Theoretical Background: How DNNs Break the Curse of Dimensionality

This document provides the mathematical foundation for the research presented in:

**"How DNNs Break the Curse of Dimensionality: Compositionality and Symmetry Learning"**  
*Arthur Jacot, Seok Hoan Choi, Yuxiao Wen*  
*ICLR 2025*

---

## Core Theoretical Contributions

### 1. Compositionality Learning Theorem

**Main Result:** Deep Neural Networks can efficiently learn any composition of functions with bounded F₁-norm, enabling them to break the curse of dimensionality.

**Formal Statement:**
Let f* = h∘g where:
- g: R^d → R^k with k << d (dimensionality reduction)
- h: R^k → R^m operates in reduced space
- Both g and h have bounded F₁-norm (Barron norm)

Then DNNs can learn f* from O(k polylog(d)) samples, avoiding the curse of dimensionality.

### 2. Generalization Bound

Our main bound combines:
- **Covering number arguments** for compositionality
- **F₁-norm control** for large width adaptivity

**Bound Formula:**
```
R(f_θ) ≤ C × (R(θ)/√n) × √(log(n))
```

Where:
- R(f_θ): Generalization error
- R(θ): Network complexity measure
- n: Sample size
- C: Universal constant

---

## Accordion Networks (AccNets)

### Architecture Design

AccNets alternate between expansion and contraction to learn compositional structure:

```
Input (d=15) → Expand (d=900) → Contract (d=100) → Expand (d=900) → ... → Output (d=20)
```

**Layer Structure:**
- **Expansion layers (in_linears)**: Learn feature representations
- **Contraction layers (out_linears)**: Perform dimensionality reduction
- **L layers**: Total number of expansion-contraction pairs

### Mathematical Motivation

AccNets are designed to approximate the composition:
```
f_L:1 = f_L ∘ f_{L-1} ∘ ... ∘ f_1
```

Where each f_i consists of:
1. Expansion: z_i → Wᵢⁱⁿ z_i (dimension increase)
2. Nonlinearity: σ(·)
3. Contraction: σ(Wᵢⁱⁿ z_i) → Wᵢᵒᵘᵗ σ(Wᵢⁱⁿ z_i) (dimension decrease)

---

## Complexity Measures

### Our Novel Complexity Bound (2024)

**Formula:**
```
R(θ) = ∏ᵢ Lᵢ × Σᵢ (||Wᵢ||_F / Lᵢ) × √(dᵢ + dᵢ₊₁)
```

**Components:**
- **∏ᵢ Lᵢ**: Product of Lipschitz constants
- **||Wᵢ||_F**: Frobenius norm of layer i
- **Lᵢ**: Lipschitz constant of layer i  
- **dᵢ**: Input dimension of layer i
- **dᵢ₊₁**: Output dimension of layer i

### Comparison with Existing Bounds

| Bound | Formula | Key Insight |
|-------|---------|------------|
| **Neyshabur 2015** | `∏ᵢ ||Wᵢ||₂` | Spectral norm product |
| **Bartlett 2017** | `Σᵢ ||Wᵢ||²_F` | Path-based analysis |
| **Neyshabur 2018** | `∏ᵢ ||Wᵢ||₂ × Σᵢ ||Wᵢ||_F` | Spectral complexity |
| **Ledent 2021** | `Σᵢ (||Wᵢ||_F × ||Wᵢ||₂)` | Path norm |
| **Ours 2024** | `∏ᵢ Lᵢ × Σᵢ (||Wᵢ||_F/Lᵢ × √dᵢ)` | **Rank-aware** |

### Rank Measures

**Standard Rank:**
```python
rank(W) = torch.linalg.matrix_rank(W, atol=0.1, rtol=0.1)
```

**Stable Rank:**
```python
stable_rank(W) = ||W||²_F / ||W||²_2
```

Stable rank provides a continuous measure of effective dimensionality.

---

## Matérn Kernels for Synthetic Data

### Mathematical Definition

**Matérn Kernel:**
```
K(x,y) = (2^(1-ν)/Γ(ν)) × (√(2ν)||x-y||/ℓ)^ν × K_ν(√(2ν)||x-y||/ℓ)
```

Where:
- **ν**: Smoothness parameter controlling differentiability
- **Γ(ν)**: Gamma function
- **K_ν**: Modified Bessel function of the second kind
- **ℓ**: Length scale (set to 1 in our experiments)

### Smoothness Levels

| ν Value | Differentiability | Kernel Type |
|---------|------------------|-------------|
| 0.5 | Non-differentiable | Exponential: `exp(-r)` |
| 1.5 | C¹ (once diff.) | `(1+√3r)exp(-√3r)` |
| 2.5 | C² (twice diff.) | `(1+√5r+5r²/3)exp(-√5r)` |
| ∞ | C∞ (smooth) | Gaussian: `exp(-r²/2)` |

### Compositional Data Generation

**Step 1:** Sample Z = g(X) using Matérn kernel with parameter ν_g
```python
kernel_g = MaternKernel(X, nu_g)
Z = kernel_g.sample((d_intermediate,))
```

**Step 2:** Sample Y = h(Z) using Matérn kernel with parameter ν_h
```python
kernel_h = MaternKernel(Z, nu_h)  
Y = kernel_h.sample((d_output,))
```

This creates compositional structure Y = h(g(X)) with controllable regularity.

---

## Key Research Questions

### 1. When Do Deep Networks Beat Shallow Networks?

**Hypothesis:** Deep networks should excel when:
- ν_g is small (g is rough, hard for shallow networks)
- ν_h is large (h is smooth, learnable in reduced space)

**Empirical Validation:** Heatmaps of test loss vs (ν_g, ν_h)

### 2. How Do AccNets Compare to Standard Deep Networks?

**AccNets Advantages:**
- Explicit bottleneck encourages dimensionality reduction
- Alternating structure mimics compositional learning
- Better parameter efficiency for compositional tasks

**Deep Networks:**
- More parameters, greater flexibility
- May learn compositions implicitly
- Established training procedures

### 3. What Role Do Complexity Bounds Play?

**Theoretical Prediction:** Networks with lower complexity bounds should generalize better.

**Empirical Test:** Correlation between computed bounds and actual test performance.

---

## Experimental Validation

### Parameter Space

**Full Experiment:**
- ν_g, ν_h ∈ [0.5, 20] with 0.5 increments (400 combinations)
- N ∈ {100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000}
- 3 architectures × 3 trials = 32,400 total experiments

**Focused Experiment:**
- ν values: {2.0, 5.0, 8.0, 10.0} (16 combinations)
- N values: {1000, 5000, 20000}
- For quick validation and debugging

### Key Findings

1. **AccNets excel** when compositional structure is present
2. **Deep networks** are competitive but require more parameters
3. **Shallow networks** suffer from curse of dimensionality
4. **Our complexity bounds** correlate well with empirical performance
5. **Symmetry learning** emerges naturally in the bottleneck representations

---

## Implications for Deep Learning Theory

### Breaking the Curse of Dimensionality

Traditional kernel methods suffer when learning functions in high dimensions. Our results show that:

1. **Compositional Structure** allows circumventing high-dimensional complexity
2. **Architectural Choices** matter: AccNets > Deep > Shallow for compositional tasks
3. **Generalization Bounds** can capture architectural benefits theoretically

### Symmetry and Compositionality

The learned function g effectively discovers **symmetries** in the data:
- g maps high-dimensional input to low-dimensional "invariant" representation
- h operates on this simplified representation
- Overall function f = h∘g exploits compositional structure

### F₁-norm and Barron Space

Functions with bounded F₁-norm can be approximated efficiently by neural networks:
```
||f||_F₁ = ∫ |f̂(ω)| ||ω|| dω < ∞
```

This connects our work to harmonic analysis and approximation theory.

---

## Connections to Related Work

### Neural Tangent Kernel (NTK) Theory
- Our bounds extend NTK analysis to compositional settings
- AccNets provide explicit architectural bias for composition

### Approximation Theory  
- F₁-norm connects to classical Barron space results
- Matérn kernels provide controlled regularity for systematic study

### Information Theory
- Bottleneck principle: optimal g preserves task-relevant information
- Rate-distortion trade-offs in learned representations

---

This theoretical foundation guides both the experimental design and the practical implementation of our compositionality learning framework.