# Main Figure Equation Mapping

Paper: Chau, Miller, and Palmigiano, "Exact linear theory of perturbation
response in a space- and feature-dependent cortical circuit model", PNAS.
DOI page: https://www.pnas.org/doi/10.1073/pnas.2426758122

This table extends the repo's figure-command mapping with the main-text and
supplementary equations that justify each plot. It also records what should be
changed when the Eq. 8 orientation factor

```text
cos(theta - phi)
```

is replaced by

```text
cos(psi - theta) cos(psi - phi).
```

In the code, the original term is implemented by `Tuning.kernel` in
`src/niarb/nn/modules/kernels.py`; the modified term is implemented by
`PsiTuning.kernel` and, for visual-field-derived `psi`, `VisualFieldTuning`.
The model selects the original or psi kernel in `V1.__init__`.

| Plot | Original plotting code | Related main-text equations | Related SI equations/proof | What the plot shows | What changes after modifying Eq. 8 |
|---|---|---|---|---|---|
| Fig. 1B | `paper/fit_kernel.py ... -o paper/figures/1b.pdf` | Eq. 2 Green's/Laplace kernel; Eq. 3 simplified spatial connectivity | SI section 1B fitting of spatial connectivity; SI data-processing notes in `paper/README.md` | Fits the empirical spatial connectivity kernel widths and amplitudes from product data. | No direct psi change. This plot does not use the orientation factor in Eq. 8. Only change it if you re-fit a new empirical model that includes psi-dependent orientation structure. |
| Fig. 1C | `paper/response.py ... -m eigvals ... -o paper/figures/1c.pdf` | Eq. 8 full connectivity; Eq. 13 discretized connectivity matrix; stability/eigenvalue discussion in main text | SI Eq. S18 full connectivity; SI Eq. S3 discretized random/finite matrix form; SI sections on spectrum/stability | Builds the finite connectivity matrix and plots eigenvalues/spectral radius. | Must use `PsiTuning`; analytic/circulant shortcuts no longer apply for independent random psi. The weight tensor must be flattened to an `N x N` dense matrix before `torch.linalg.eigvals`. |
| Fig. 1D left | `paper/response.py ... -m compare -k space_ori ... -o paper/figures/1d_a.pdf` | Eq. 1 recurrent response equation; Eq. 4 `(I-W)r=h`; Eq. 8 connectivity; Eq. 9 exact response; Eq. 14 discretized analytic response | SI Eqs. S18-S25 orientation/Fourier decomposition; SI Eqs. S47-S50 exact response kernel; Methods "Comparison of theory and simulations" | Compares analytic theory, numerical simulation, and matrix approximation for space-orientation responses. | Old "Theory" curve depends on separability of `cos(theta-phi)`. For psi, replace it with finite `Matrix` reference. Use dense matrix solve and fixed `--seed`. |
| Fig. 1D right | `paper/response.py ... -m compare -k ori_osi ... --tau-i 1.0 ... -o paper/figures/1d_b.pdf` | Same as Fig. 1D left, with OSI/selectivity dependence in Eq. 8 | SI Eq. S18 includes `f_alpha(mu) g_beta(nu)`; SI Eq. S50 exact feature-dependent kernel; SI feature/OSI integration discussion | Compares response as a function of orientation and OSI/selectivity. | Same as 1D left. The psi term changes orientation dependence, but OSI factors remain unchanged unless you also alter `f`/`g` or OSI distribution. |
| Fig. 2A | `paper/contourplot_space.py E ... -o paper/figures/2a.pdf` | Eq. 2 Green's kernel; Eq. 3 spatial connectivity; Eq. 7/10 Green's-function mixture for spatial response | SI zero-crossing proof, especially SI Eq. S1 and related root-finding conditions; SI sections deriving spatial response mixtures | Phase diagram for number of excitatory-response zero crossings when E/I spatial kernels have equal width. | Do not simply add `--use-psi`. This contour is from analytic spatial theory. A psi version requires re-deriving the contour conditions after replacing the orientation operator. |
| Fig. 2B / 5B | `paper/contourplot_space.py E --rho 1 2 ...` | Same spatial-response theory as Fig. 2A, now varying width ratio `rho` | Same SI zero-crossing and spatial-response proof as Fig. 2A | Shows how unequal E/I spatial widths affect spatial-response regimes. | Same as 2A: requires new analytic theory if psi should be included. |
| Fig. 2C | `paper/response.py ... -m space --normalize ... -o paper/figures/2c.pdf` | Eq. 1; Eq. 4; Eq. 8; Eq. 12 numerical dynamics; Eq. 14 finite-volume response | SI Eq. S18 full model; SI S47-S50 response kernel; Methods comparison of theory and simulation | Simulated/model spatial responses for selected connectivity regimes. | Can be run with psi, but the reference must be matrix/numerical rather than old analytic theory. If using independent psi, use a smaller grid or high-memory dense solve. |
| Fig. 2D | `paper/contourplot_space.py r0 ... -o paper/figures/2d.pdf` | Spatial zero-crossing analysis from Eq. 7/10 | SI Eq. S1 root condition; SI bootstrap details for confidence interval from perturbation data | Contour for first zero crossing `r0`, with empirical confidence shading. | Not automatically changed by psi. The plotted contour assumes the original analytic spatial response. |
| Fig. 2E | `paper/contourplot_space.py rmin ... -o paper/figures/2e.pdf` | Spatial response extrema from Eq. 7/10 | SI Eq. S1 and extrema/root derivations; SI bootstrap details for `s_min - s0` interval | Contour for minimum-response location relative to the zero crossing. | Needs re-derived psi response before a psi contour is meaningful. |
| Fig. 2F | `paper/contourplot_space.py dr1dw11 ... -o paper/figures/2f.pdf` | Sensitivity of spatial response to connection weights, derived from Eq. 7/10 | SI spatial-response derivative/sensitivity derivations | Shows derivative of a response feature with respect to recurrent weight. | Re-derive derivative for the psi-modified operator; old derivative is not valid by substitution. |
| Fig. 2G | `paper/contourplot_space.py decay ... -o paper/figures/2g.pdf` | Green's-function decay and eigenmode mixture, Eq. 2 and Eq. 7/10 | SI mixture/eigenvalue proof and decay analysis | Shows spatial decay regimes of the exact response. | Only changes if the psi-modified operator changes the eigenmode mixture being analyzed. For independent random psi, the old closed-form decay plot is not applicable. |
| Fig. 3A | `paper/contourplot_space.py EI ... -o paper/figures/3a.pdf` | Eq. 7/10 spatial response for E and I populations | SI section relating E and I zero crossings; inequalities around spatial eigenvalues | Predicts when inhibitory responses share or differ from excitatory zero-crossing structure. | This is analytic spatial theory. Do not label as psi-modified without re-deriving the E/I inequalities. |
| Fig. 3B | `paper/contourplot_space.py rEI ... -o paper/figures/3b.pdf` | Mean response constraints and spatial response mixture | SI mean response and E/I response-sign derivations | Relates mean inhibitory and excitatory response structure. | Requires re-derivation if psi changes the feature-averaged operator. If averaging over independent uniform psi, the expected orientation term is effectively `0.5 cos(theta-phi)`, but a sampled network is not the same as simply halving `kappa`. |
| Fig. 4A | `paper/response.py ... -m ori ... -o paper/figures/4a.pdf` | Eq. 8 orientation-tuned connectivity; Eq. 9 exact response as untuned plus cosine-tuned part | SI Eq. S18 full connectivity; SI Eqs. S47-S54 feature/orientation response and feature-integrated response | Orientation-only perturbation response: same-favoring vs opposite-favoring. | Good psi target. Replace `Tuning` by `PsiTuning`, use matrix reference, keep OSI factors unchanged, and seed independent psi sampling. |
| Fig. 4B / 5D | `paper/contourplot_ori.py ... -o paper/figures/4b.pdf` | Orientation-response sign/transition analysis from Eq. 8/9 | SI Eq. S50 and SI Eq. S54 after integrating over space; proof of same- vs opposite-favoring regimes | Analytic phase diagram for same-favoring vs opposite-favoring orientation response. | Needs new analytic orientation theory. The original proof uses the Fourier/cosine decomposition of `cos(theta-phi)`; independent psi breaks that exact decomposition for a finite sampled network. |
| Fig. 4C | `paper/response.py ... -m space_ori --normalize ... -o paper/figures/4c.pdf` | Eq. 8 connectivity; Eq. 9 exact space-feature response; Eq. 12 numerical simulation | SI Eq. S18; SI Eq. S50; Methods comparison of theory and simulations | Space-orientation response showing distance and tuning-preference effects. | Good psi target. Use finite matrix/numerical response, label reference as `Matrix`, avoid analytic theory, flatten dense tensors as needed, and avoid seaborn relplot wrapper issues. |
| Fig. 4D | `paper/contourplot_space.py E ... --ori ... -o paper/figures/4d.pdf` | Combined spatial/orientation analytic phase structure from Eq. 8/9 | SI Eq. S50 and feature-integrated response analysis | Analytic contour for feature-dependent spatial response. | Needs re-derived psi contour. Do not treat old contour as psi-modified by changing only code flags. |
| Fig. 5A | `paper/mean_response_gain.py ... -o paper/figures/5a.pdf` | Eq. 12 dynamical/gain response; mean response discussion after Eq. 9 | SI Eq. S9 gain modulation; SI Eqs. S47-S56 mean and feature-integrated response proofs | Shows how changing neuronal gain modulates mean response. | If psi is included, gain modulation should be recomputed with the modified finite operator. Existing analytic mean-response proof may not hold without re-derivation. |
| Fig. 5C | `paper/contourplot_space.py dr0dg ... -o paper/figures/5c.pdf` | Gain derivative of spatial zero-crossing structure | SI Eq. S9 and derivative/zero-crossing proof | Shows how zero crossings change with gain. | Needs re-derived derivative under the psi-modified operator. |
| Fig. 6A-H | `niarb plot ...` after fitted `fit.toml` and `run*.toml` workflows | Eq. 8 full model; Eq. 12 simulation dynamics; Eq. 13 finite/disordered connectivity; model fitting objective | SI Eq. S2 fitting objective; SI Eq. S3 disordered finite matrix; SI S4-S8 sparse probability/strength kernels; model-fit README workflows | Fits model parameters to perturbation data and plots fitted parameter distributions and responses. | Plotting alone is insufficient. To make true psi versions, pass `use_psi=true` into the model configs used for fitting/running, rerun fits/runs, then plot those new outputs. |

## Practical consequence of the psi replacement

The original proof exploits that

```text
cos(theta - phi)
```

is a translation-invariant kernel on the orientation circle. This permits a
Fourier-mode decomposition: an untuned mode plus a tuned cosine mode. That is
why the paper can derive exact analytic response expressions such as the
main-text Eq. 9 and the SI Eq. S50.

The replacement

```text
cos(psi - theta) cos(psi - phi)
```

has two cases:

1. If `psi` is an independent random variable per connection, the finite
   sampled network is not circulant in orientation. The old analytic response
   and contour proofs do not apply directly; use dense finite matrices.
2. If averaging analytically over independent uniform `psi`,

   ```text
   E_psi[cos(psi - theta) cos(psi - phi)] = 1/2 cos(theta - phi),
   ```

   so the average kernel resembles the original model with half the orientation
   modulation. This is not equivalent to one sampled random-psi network.
3. If `psi` is computed from a visual-field map, it couples cortical position
   and orientation. This breaks the old separation between space and feature
   variables and requires a new proof before contour/theory figures can be
   interpreted as exact analytic results.

