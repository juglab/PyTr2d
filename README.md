# PyTr2d

PyTr2d is a Python project for 2D cell tracking on Cell Tracking Challenge style data, with support for:

- learning event costs from annotated training data,
- single-source tracking on one segmentation source at a time,
- resumable tracking runs with checkpoints,
- a consensus mode that merges two saved tracking results,
- and Cell Tracking Challenge style outputs (`maskNNN.tif` and `res_track.txt`).

The current code is designed around the `Fluo-N2DL-HeLa` dataset layout and uses:

- sequence `01` for training the event classifiers,
- sequence `02` for tracking,
- Gurobi for optimization,
- random forests for event scoring,
- optional official Cell Tracking Challenge evaluation through `py-ctcmetrics`.

## What The Project Does

PyTr2d currently has two tracking modes.

### 1. `single` mode

This is the default mode.

- It loads one segmentation source, or several sources if `--seg-source all` is used.
- It trains or loads four event classifiers:
  - `move`
  - `division`
  - `appearance`
  - `disappearance`
- It then performs **pairwise ILP tracking** between consecutive frames:
  - `(t, t+1)` for all frames in sequence `02`
- It writes one tracked mask per frame and one lineage file.
- It can also evaluate an already saved single-source result without retracking by using `--evaluate-only`.

### 2. `consensus` mode

This mode merges two already tracked external solutions, by default:

- `embedseg`
- `stardist`

It works as follows:

- If the saved `embedseg` or `stardist` tracking result is missing, it first runs the missing single-source tracking job.
- It loads both saved tracking solutions.
- It finds shared trajectory fragments using one-to-one frame matching with `IoU >= 0.8` and matching temporal structure.
- These shared fragments are treated as **common-supported tracklets**.
- The remaining non-shared fragments become hypotheses.
- It solves one **joint tracklet-and-geometry ILP** over the whole sequence.
- The joint ILP includes both tracklet-selection variables and common-fragment segmentation-choice variables.
- It exports one optimized result:
  - `optimized_joint`

Source-specific selected fragments always keep the geometry from their own source. Common-supported selected fragments share the same lineage fragment but still need a geometry choice, because both sources provide a valid mask for that fragment.

### 3. `--evaluate-only`

This is not a separate `--mode`. It is a flag that changes what `single` or `consensus` mode does.

- In `single` mode, it loads one already saved source result and recomputes only the metric files.
- In `consensus` mode, it loads the already saved consensus outputs and recomputes only the pre-merge, per-variant, and comparison metric files.
- When `render_manifest.json` is present, it uses that manifest to decide which consensus variants to reevaluate.
- It skips:
  - classifier loading and training,
  - pairwise tracking ILPs,
  - consensus/global ILP solving.

## Expected Data Layout

The code expects a dataset layout like this:

```text
data/
  Fluo-N2DL-HeLa_train/
    Fluo-N2DL-HeLa/
      01/
      01_ERR_SEG/
      01_GT/
        TRA/
          man_track.txt
          *.tif
      01_ST/
        SEG/
      02/
      02_ERR_SEG/
      02_GT/
        TRA/
      02_ST/
        SEG/
    Segmentations/
      embedseg/
      stardist/
```

### Meaning Of The Folders

- `01/`
  - raw training frames
- `02/`
  - raw tracking frames
- `01_GT/TRA/`
  - training ground-truth masks and `man_track.txt`
- `02_GT/TRA/`
  - optional evaluation ground truth for sequence `02`
- `01_ST/SEG/`, `02_ST/SEG/`
  - CTC `ST` segmentation folders
- `01_ERR_SEG/`, `02_ERR_SEG/`
  - CTC `ERR_SEG` segmentation folders
- `Segmentations/embedseg/`, `Segmentations/stardist/`
  - extra external segmentations used as tracking sources

### Built-In Source Names Used By The Code

- `st`
  - maps to `<sequence>_ST/SEG`
- `err_seg`
  - maps to `<sequence>_ERR_SEG`
- any folder name under `Segmentations/`
  - for example `embedseg`, `stardist`

By default, `--dataset-root` points to:

```text
./data/Fluo-N2DL-HeLa_train/Fluo-N2DL-HeLa
```

and `--extra-seg-root` defaults to:

```text
./data/Fluo-N2DL-HeLa_train/Segmentations
```

if that folder exists.

## Installation

Core runtime dependencies are:

- `numpy`
- `scipy`
- `gurobipy`
- `scikit-image`
- `scikit-learn`
- `tifffile`
- `tqdm`
- `py-ctcmetrics`

Install them in your preferred environment, for example with `uv` or `pip`.

Example:

```bash
uv pip install -e .
```

or:

```bash
pip install -e .
```

You also need a working Gurobi installation and license.

## Command Line Interface

The main entrypoint is:

```bash
uv run python main.py
```

If you want to clean an external segmentation folder before tracking, you can also run:

```bash
uv run python normalize_instance_masks.py \
  --input-dir data/Fluo-N2DL-HeLa_train/Segmentations/embedseg \
  --output-dir data/Fluo-N2DL-HeLa_train/Segmentations/embedseg_clean
```

This script splits disconnected components that accidentally share the same label id inside a frame and writes a cleaned instance mask for each frame. By default it relabels each frame sequentially from `1..N`, which is often easier to inspect before tracking.

### Important CLI Flags

- `--mode {single,consensus}`
  - choose the tracking mode
- `--dataset-root`
  - dataset folder containing `01`, `02`, `*_GT`, `*_ST`, and `*_ERR_SEG`
- `--extra-seg-root`
  - external segmentation folder such as `Segmentations`
- `--train-sequence`
  - defaults to `01`
- `--track-sequence`
  - defaults to `02`
- `--seg-source`
  - used in `single` mode, defaults to `all`
- `--consensus-sources SOURCE_1 SOURCE_2`
  - used in `consensus` mode, defaults to `embedseg stardist`
- `--evaluate-only`
  - skip tracking and recompute metrics only from saved outputs
- `--agreement-iou-threshold`
  - used in `consensus` mode, defaults to `0.8`
- `--common-geometry-mode {posthoc,two_stage,joint}`
  - compatibility flag in `consensus` mode; the code now always normalizes this to `joint`
- `--geometry-source-weight`
  - weight of source-consistency agreement in optimized common-fragment geometry selection, defaults to `1.0`
- `--geometry-temporal-overlap-weight`
  - weight of temporal boundary overlap in optimized common-fragment geometry selection, defaults to `0.25`
- `--geometry-neighbor-radius`
  - radius used to build same-frame geometry neighbors in optimized common-fragment geometry selection, defaults to `5`
- `--max-distance`
  - maximum centroid distance for move and division candidates, defaults to `50`
- `--output-dir`
  - output directory for `single` mode
- `--consensus-output-dir`
  - output directory for `consensus` mode
- `--model-dir`
  - root directory for saved classifier bundles
- `--force-retrain`
  - ignore any saved classifier bundle and retrain the four random forests
- `--force-retrack`
  - ignore saved tracking checkpoints and rebuild tracking from frame `0`
- `--log-file`
  - path to the run log file
- `--log-level {DEBUG,INFO,WARNING,ERROR}`
  - terminal logging level

## Logging

Every run logs to:

- the terminal
- and a logfile on disk

By default:

- in `single` mode:
  - `<output-dir>/run.log`
- in `consensus` mode:
  - `<consensus-output-dir>/run.log`

Use:

```bash
--log-level DEBUG
```

for more verbose logging.

Gurobi is also configured to log to the terminal and to the same logfile.

## Classifier Training And Reuse

PyTr2d uses four random-forest classifiers:

- `move`
- `division`
- `appearance`
- `disappearance`

They are trained from:

- the raw images in sequence `01`
- the segmentation sources available for training in sequence `01`
- the tracking GT in `01_GT/TRA`

### Default Model Bundle Location

Saved classifier bundles are written to:

```text
models/<dataset-name>/<train-sequence>/event_scorers.pkl
```

For the default dataset and training sequence, that means:

```text
models/Fluo-N2DL-HeLa/01/event_scorers.pkl
```

### Bundle Reuse Rules

On each run:

- if a compatible bundle exists, it is loaded,
- if it is missing, the classifiers are trained and saved,
- if metadata does not match the current configuration, the classifiers are retrained,
- if `--force-retrain` is given, the classifiers are always rebuilt.

The saved metadata includes:

- dataset name
- training sequence
- feature version
- `max_distance`
- training IoU threshold
- random-forest hyperparameters

## Single-Mode Tracking

### Example: Track Only `stardist`

```bash
uv run python main.py \
  --mode single \
  --dataset-root ./data/Fluo-N2DL-HeLa_train/Fluo-N2DL-HeLa \
  --extra-seg-root ./data/Fluo-N2DL-HeLa_train/Segmentations \
  --seg-source stardist
```

### Example: Track Only `embedseg`

```bash
uv run python main.py --mode single --seg-source embedseg
```

### Example: Track All Available Sources Together

```bash
uv run python main.py --mode single --seg-source all
```

### Example: Evaluate Only A Saved `stardist` Result

```bash
uv run python main.py --seg-source stardist --evaluate-only
```

This means:

- load the saved single-source result from `outputs/Fluo-N2DL-HeLa/02/stardist/`
- do not retrack the sequence
- recompute and rewrite `metrics.json` and `metrics.txt`

### Example: Evaluate Only A Saved `embedseg` Result

```bash
uv run python main.py --seg-source embedseg --evaluate-only
```

This is the same evaluation-only workflow, but for the saved `embedseg` output directory:

```text
outputs/Fluo-N2DL-HeLa/02/embedseg/
```

### How Single Mode Works

- Frame `0` is initialized from the selected segmentation source.
- If multiple sources are selected, frame `0` is initialized by a small overlap-aware ILP.
- Each following frame pair is solved by a pairwise ILP.
- The code supports:
  - movement
  - division
  - appearance
  - disappearance
- Tracking checkpoints are saved after every completed frame.

### Resume Behavior

Single mode writes:

- `maskNNN.tif`
- `res_track.txt`
- `tracking_checkpoint.json`

If a run is interrupted:

- the next run resumes from the last completed frame,
- unless `--force-retrack` is used.

## Consensus Mode

### Example: Merge `embedseg` And `stardist`

```bash
uv run python main.py \
  --mode consensus \
  --dataset-root ./data/Fluo-N2DL-HeLa_train/Fluo-N2DL-HeLa \
  --extra-seg-root ./data/Fluo-N2DL-HeLa_train/Segmentations \
  --consensus-sources embedseg stardist
```

### Example: Merge With Joint Lineage And Geometry Optimization

```bash
uv run python main.py \
  --mode consensus \
  --dataset-root ./data/Fluo-N2DL-HeLa_train/Fluo-N2DL-HeLa \
  --extra-seg-root ./data/Fluo-N2DL-HeLa_train/Segmentations \
  --consensus-sources embedseg stardist
```

### What Consensus Mode Uses

Consensus mode only uses the two named **external** source results as inputs.

It does **not** use:

- `GT`
- `ST`
- `ERR_SEG`

as source solutions for the merge.

### How Consensus Mode Works

1. Check whether the two saved source-specific tracking results already exist.
2. If one is missing or incomplete, run that single-source tracker first.
3. Load both saved tracked mask stacks and lineage files.
4. Match objects frame by frame with one-to-one IoU matching.
5. Keep only matched object pairs with `IoU >= 0.8` by default.
6. Build common tracklets from shared trajectory fragments.
7. Treat these common-supported tracklets as shared lineage candidates.
8. Convert the remaining fragments from both solutions into hypothesis tracklets.
9. Solve one enlarged ILP that chooses lineage structure and common-fragment geometry together.
10. Export the optimized result as `optimized_joint/`.

### Consensus Output Variants

For source-specific selected fragments, the geometry always comes from the source that generated that fragment.

For selected common-supported fragments, the joint ILP chooses one of:

- `embedseg`
- `stardist`
- `intersection`
- `union`

The optimized geometry objective uses:

- source-consistency agreement with neighboring fragments
- a temporal boundary-overlap bonus on selected move and division relations

## Output Conventions

### Single Mode Outputs

By default:

```text
outputs/<dataset-name>/<track-sequence>/<seg-source>/
```

Example:

```text
outputs/Fluo-N2DL-HeLa/02/stardist/
```

Contains:

- `mask000.tif`, `mask001.tif`, ...
- `res_track.txt`
- `metrics.json`
- `metrics.txt`
- `tracking_checkpoint.json`
- `run.log`

### Consensus Mode Outputs

By default:

```text
outputs/<dataset-name>/<track-sequence>/consensus_<source1>_<source2>/
```

Example:

```text
outputs/Fluo-N2DL-HeLa/02/consensus_embedseg_stardist/
```

Contains:

- `run.log`
- `premerge_metrics.json`
- `premerge_metrics.txt`
- `variant_comparison.json`
- `variant_comparison.txt`
- `render_manifest.json`
- `optimized_joint/`
- `geometry_assignments.json`

Each variant folder contains:

- `mask000.tif`, `mask001.tif`, ...
- `res_track.txt`
- `metrics.json`
- `metrics.txt`

## Evaluation Metrics

PyTr2d reports metrics in both `single` and `consensus` mode.

### Official Cell Tracking Challenge Evaluation

If `02_GT/TRA` exists and `ctc_evaluate` from `py-ctcmetrics` is available in the environment, PyTr2d also runs the official CTC-style evaluation after tracking or during `--evaluate-only`.

The saved `ctc_evaluation` block can contain:

- `Valid`
- `DET`
- `SEG`
- `TRA`
- `LNK`
- `CT`
- `TF`
- `BC(0)`
- `CCA`
- derived `BIO`
- derived `OP_CSB`
- derived `OP_CTB`
- derived `OP_CLB`

If the tool is not installed, the code still writes the metrics files, but the `ctc_evaluation` block is marked as `skipped` with the reason.

### Single-Mode Metrics

For a saved single-source result, PyTr2d writes:

- `summary_metrics`
  - number of tracks
  - number of divisions
  - number of frames
  - frame object count min/mean/max
- `ctc_evaluation`
  - official CTC metrics when available

### 1. Pre-Merge Metrics

These compare the two input tracking solutions before consensus merging.

Agreement metrics:

- object precision
- object recall
- object F1
- move precision
- move recall
- move F1
- division precision
- division recall
- division F1
- shared tracklet coverage

If `02_GT/TRA` exists, PyTr2d also evaluates each input solution against GT using:

- vertex precision
- vertex recall
- vertex F1
- link precision
- link recall
- link F1
- `CT`
  - complete tracks
- `TF`
  - track fractions
- `BC(0)`
  - branching correctness with zero-frame tolerance
- `BIO`
  - mean of `CT`, `TF`, and `BC(0)`

In addition, each input solution now also gets a nested `ctc_evaluation` block in `premerge_metrics.*` when official CTC evaluation is available.

### 2. Final Variant Metrics

For each rendered final variant, PyTr2d writes:

- `summary_metrics`
  - number of tracks
  - number of divisions
  - number of frames
  - frame object count min/mean/max
  - common tracklet coverage
- `legacy_gt_metrics` if `02_GT` exists:
  - vertex precision
  - vertex recall
  - vertex F1
  - link precision
  - link recall
  - link F1
  - `CT`
  - `TF`
  - `BC(0)`
  - `BIO`
- `ctc_evaluation`
  - official CTC metrics when available

For optimized geometry modes, each variant `metrics.*` file also includes an `optimized_geometry` block with:

- counts by chosen geometry option
- average source-consistency score
- average temporal overlap bonus
- the configured neighbor radius and geometry weights
- the number of common-fragment geometry options ruled out by conflicts

The root-level `variant_comparison.*` files compare whichever rendered variants are present for the current run.

## Cell Tracking Challenge Output Format

The final outputs follow the usual CTC-style convention:

- one tracked mask per frame:
  - `mask000.tif`, `mask001.tif`, ...
- one lineage file:
  - `res_track.txt`

Each row of `res_track.txt` has:

```text
track_id begin end parent
```

where:

- `track_id`
  - final track identifier
- `begin`
  - first frame index of the track
- `end`
  - last frame index of the track
- `parent`
  - parent track id, or `0` if the track has no parent

## Typical Workflows

### Train Classifiers And Track One Source

```bash
uv run python main.py --mode single --seg-source stardist
```

### Force Rebuild Of The Classifier Bundle

```bash
uv run python main.py --mode single --seg-source stardist --force-retrain
```

### Resume A Previous Single-Source Tracking Run

```bash
uv run python main.py --mode single --seg-source embedseg
```

If a valid checkpoint exists, it resumes automatically.

### Restart Tracking From Scratch

```bash
uv run python main.py --mode single --seg-source embedseg --force-retrack
```

### Run The Full Consensus Merge

```bash
uv run python main.py --mode consensus --consensus-sources embedseg stardist
```

### Evaluate Only A Saved Consensus Result

```bash
uv run python main.py --mode consensus --consensus-sources embedseg stardist --evaluate-only
```

This means:

- load the saved input source results from:
  - `outputs/Fluo-N2DL-HeLa/02/embedseg/`
  - `outputs/Fluo-N2DL-HeLa/02/stardist/`
- load the saved consensus variants from:
  - `outputs/Fluo-N2DL-HeLa/02/consensus_embedseg_stardist/`
- if `render_manifest.json` exists, use it to discover the saved variant names
- do not rerun pairwise tracking
- do not rerun the consensus/global ILP
- recompute and rewrite:
  - `premerge_metrics.json`
  - `premerge_metrics.txt`
  - each variant `metrics.json`
  - each variant `metrics.txt`
  - `variant_comparison.json`
  - `variant_comparison.txt`

### When To Use `--seg-source` In Evaluation-Only Mode

`--seg-source` matters only in `single` mode.

Examples:

```bash
uv run python main.py --seg-source stardist --evaluate-only
uv run python main.py --seg-source embedseg --evaluate-only
```

These commands tell PyTr2d which saved single-source result directory to load:

- `stardist` means:
  - `outputs/Fluo-N2DL-HeLa/02/stardist/`
- `embedseg` means:
  - `outputs/Fluo-N2DL-HeLa/02/embedseg/`

In `consensus` mode, `--seg-source` is not used. The relevant inputs are chosen through:

```bash
--consensus-sources embedseg stardist
```

## Testing

Run the unit tests with:

```bash
uv run python -m unittest discover -s tests -v
```

The dataset smoke tests are opt-in:

```bash
PYTR2D_RUN_SMOKE=1 uv run python -m unittest discover -s tests -v
```

## Notes And Current Scope

- The code is currently focused on 2D tracking.
- The main target layout is `Fluo-N2DL-HeLa`.
- Event scorers are trained on sequence `01` and tracking is run on sequence `02`.
- `single` mode uses pairwise frame-to-frame ILPs.
- `consensus` mode uses a global tracklet-level ILP on top of two saved single-source tracking results.
- The project is designed to log heavily to both terminal and file for long runs.

## Repository Entry Points

- [main.py](main.py)
  - CLI entrypoint and orchestration
- [dataio/projectio.py](dataio/projectio.py)
  - data loading, saved-result loading, output writing
- [tracking/random_forest.py](tracking/random_forest.py)
  - event features and classifier training/loading
- [tracking/trackingsolver.py](tracking/trackingsolver.py)
  - single-mode pairwise ILP tracker
- [tracking/consensus.py](tracking/consensus.py)
  - consensus preparation, global tracklet ILP, variant rendering, metrics
