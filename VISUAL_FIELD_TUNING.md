# Visual-field tuning modification

This branch adds an optional replacement for the Eq. 8 orientation term. The
default is still the original model.

Original Eq. 8 orientation factor:

```text
1 + 2 kappa_ab cos(theta - phi)
```

New optional factor, enabled only with `use_psi = true`:

```text
1 + 2 kappa_ab cos(psi - theta) cos(psi - phi)
```

If `use_visual_field_tuning = true`, `psi` is computed pairwise from the vector
between the two neurons' receptive-field centers in visual-field coordinates.
The implementation maps each neuron's existing repo `space` coordinate to a 2D
visual-field coordinate, then uses:

```text
psi = atan2(rf_pre_y - rf_post_y, rf_pre_x - rf_post_x)
```

If `use_visual_field_tuning = false`, `psi` is sampled as an independent uniform
random angle. This is the default psi mode. You may optionally provide a fixed
`psi` value in the same coordinate system as `ori` for debugging or controlled
comparisons.

## Files changed

- `src/niarb/atlas.py`: affine V1-to-visual-field map loader.
- `src/niarb/nn/modules/kernels.py`: new `VisualFieldTuning` kernel.
- `src/niarb/nn/modules/v1.py`: optional `visual_field_map` argument on `nn.V1`.
- `scripts/check_visual_field_tuning.py`: quick smoke-test script.
- `scripts/fit_allen_visual_field_map.py`: fits the JSON map from point correspondences.
- `examples/allen_visual_field_map.example.json`: example mapping file format.

## Important limitation

The Allen atlas data itself is not present in this repository. The new code
therefore expects you to provide a local affine mapping file derived from Allen
atlas/retinotopy registration. The file must use the same input coordinate
system as this repo's `space` variable.

## Mapping file format

Create a JSON file like:

```json
{
  "matrix": [[a00, a01], [a10, a11]],
  "offset": [b0, b1]
}
```

This computes:

```text
rf_x = a00 * space_x + a01 * space_y + b0
rf_y = a10 * space_x + a11 * space_y + b1
```

A CSV file with one row is also accepted if it has columns:

```text
matrix_00,matrix_01,matrix_10,matrix_11,offset_0,offset_1
```

## Creating the map from Allen-derived points

Export a CSV of matched points from your Allen atlas/retinotopy workflow:

```text
cortical_x,cortical_y,visual_x,visual_y
0.0,0.0,-20.0,10.0
100.0,0.0,-18.5,10.2
0.0,100.0,-20.3,12.1
```

Then fit the affine map:

```bash
python scripts/fit_allen_visual_field_map.py \
  path/to/allen_correspondence_points.csv \
  -o path/to/allen_visual_field_map.json
```

## Quick checks

From the repo root:

```bash
python scripts/check_visual_field_tuning.py
```

Use psi as an independent variable, without visual-field tuning:

```bash
python scripts/check_visual_field_tuning.py --use-psi --seed 0
```

Use a fixed independent psi value for debugging:

```bash
python scripts/check_visual_field_tuning.py --use-psi --psi 45
```

Use psi computed from visual-field RF positions:

```bash
python scripts/check_visual_field_tuning.py \
  --use-psi \
  --use-visual-field-tuning \
  --map examples/allen_visual_field_map.example.json
```

## Using it in Python

```python
from niarb import atlas, nn

visual_map = atlas.AllenAffineVisualFieldMap("path/to/allen_visual_field_map.json")
model = nn.V1(
    ["cell_type", "space", "ori"],
    cell_types=["PYR", "PV"],
    use_psi=True,
    use_visual_field_tuning=True,
    visual_field_map=visual_map,
    mode="matrix",
)
```

To use psi but not visual-field tuning:

```python
from niarb import nn

model = nn.V1(
    ["cell_type", "space", "ori"],
    cell_types=["PYR", "PV"],
    use_psi=True,
    use_visual_field_tuning=False,
    mode="matrix",
)
```

Use `mode="matrix"`, `mode="matrix_approx"`, or `mode="numerical"` for response
simulations. The original `mode="analytical"` response derivation assumes the
old separable `cos(theta - phi)` orientation term and is not valid for this
space-orientation-coupled modification.

## Using it in TOML configs

Add this to a model config:

```toml
[pipeline.model]
mode = "matrix"
use_psi = true
use_visual_field_tuning = true

[pipeline.model.visual_field_map]
__call__ = ["niarb.atlas", "AllenAffineVisualFieldMap"]
path = "path/to/allen_visual_field_map.json"
```

To use psi but not visual-field tuning, do not provide a visual map:

```toml
[pipeline.model]
mode = "matrix"
use_psi = true
use_visual_field_tuning = false
```

Optionally add `psi = 45.0` for a fixed independent psi value instead of
uniform random sampling.

or use `mode = "numerical"` if you need nonlinear dynamics.

## Running `paper/response.py` with psi but not visual-field tuning

The paper response script now exposes the same switches. For example:

```bash
python paper/response.py \
  --mode space_ori \
  --wee 1.5 1.5 \
  --wei 3 3 \
  --wie 3 3 \
  --wii 5 5 \
  --kee 0.15 0.15 \
  --kei 0.5 0.3 \
  --kie 0.4 0.15 \
  --kii 0.5 0.5 \
  --N-space 40 40 \
  --N-ori 12 \
  --use-psi \
  --seed 0 \
  --dh 10000 \
  --out paper/figures/psi_no_visual_field.pdf
```

Do not pass `--use-visual-field-tuning` and do not pass `--visual-field-map`.
With `--use-psi` alone, psi is independent and uniformly sampled.
Passing `--seed` makes that independent psi sampling reproducible.

Psi runs are dense because a random independent `psi` breaks the old circulant
space/orientation shortcut. For an interactive smoke test, use a smaller grid:

```bash
python paper/response.py \
  --mode space_ori \
  --wee 1.5 1.5 \
  --wei 3 3 \
  --wie 3 3 \
  --wii 5 5 \
  --kee 0.15 0.15 \
  --kei 0.5 0.3 \
  --kie 0.4 0.15 \
  --kii 0.5 0.5 \
  --N-space 8 8 \
  --N-ori 6 \
  --use-psi \
  --seed 0 \
  --dh 10000 \
  --out paper/figures/psi_no_visual_field_smoke.pdf
```

The full `40 x 40 x 12` psi example has 38,400 neurons and needs dense matrices
with tens of GiB of memory. Run that only as a batch job on a large-memory
compute node and raise `--max-neurons` deliberately.

For the psi path, the script uses the finite matrix solve for the reference
curve because the old analytic response assumes the original `cos(theta - phi)`
term.
