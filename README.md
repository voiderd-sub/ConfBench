# ConfBench: Conformational Change Benchmark

**ConfBench** evaluates how well structure prediction models can predict holo structures from apo structures.

> **Note**: This implementation is a modified version of ConfBench proposed in [NeuralPlexer3](https://arxiv.org/abs/2412.10743).
> Additional filters: **resolution-based filter**, **lddt-based filter**

---

## Score Definition

$$Score = \frac{RMSD(Pred, Apo) - RMSD(Pred, Holo)}{\sqrt{\frac{1}{2} \left( RMSD(Pred, Apo)^2 + RMSD(Pred, Holo)^2 + RMSD(Apo, Holo)^2 \right)}}$$

Three scores are computed for each data point:
- **Global Score**: Based on all CA atoms
- **Pocket CA Score**: Based on pocket CA atoms
- **Pocket All Score**: Based on all pocket atoms (non-H)

| Score | Interpretation |
|-------|----------------|
| > 0 | Prediction is closer to Holo (good) |
| = 0 | Prediction is between Apo and Holo |
| < 0 | Prediction is closer to Apo (bad) |

---

## Installation

```bash
pip install pandas numpy biopython tqdm pymol-open-source pyarrow
```

---

## Usage

### Step 1: Calculate RMSD for all pairs

Calculate RMSD (Global, Pocket CA, Pocket All) for all holo-apo pairs in the PLINDER dataset.

```bash
python calculate_rmsd.py \
    --plinder-base /path/to/plinder \
    --systems-dir /path/to/plinder/systems \
    --linked-structures-dir /path/to/plinder/linked_structures \
    --links-file /path/to/links.parquet \
    --annotation-table /path/to/annotation_table.parquet \
    --pocket-distance-cutoff 10.0 \
    --output-dir results
```

### Step 2: Prepare ConfBench data

Copy structure files for all successful pairs into a standalone `confbench_data` directory.

```bash
python prepare_confbench_data.py \
    --rmsd-csv results/rmsd_results_YYYYMMDD_HHMMSS.csv \
    --links-file /path/to/links.parquet \
    --systems-dir /path/to/plinder/systems \
    --linked-structures-dir /path/to/plinder/linked_structures \
    --output-dir confbench_data
```

Output structure:
```
confbench_data/
├── metadata.csv
├── holo/{holo_id}/receptor.cif, ligand_files/*.sdf
└── apo/{apo_id}.cif
```

### Step 3: Run your model

Generate predicted structures using the apo structures in `confbench_data/apo/`.
Save predictions in a directory with filenames like `{holo_id}.cif` or `{holo_id}-*.cif`.

### Step 4: Run benchmark

Evaluate the predictions. Filtering options can be applied here.

```bash
python run_benchmark.py \
    --confbench-data confbench_data \
    --predictions-dir /path/to/predictions \
    --output-csv benchmark_results.csv \
    --lddt-threshold 0.6 \
    --max-holo-resolution 4.5 \
    --max-apo-resolution 4.5 \
    --min-rmsd 1.5
```

---

## Filtering Options

| Option | Default | Description |
|--------|---------|-------------|
| `--lddt-threshold` | 0.6 | Minimum lddt |
| `--max-holo-resolution` | 4.5Å | Maximum holo resolution |
| `--max-apo-resolution` | 4.5Å | Maximum apo resolution |
| `--min-rmsd` | 1.5Å | At least one RMSD must exceed |

To disable filters: `--no-lddt-filter`, `--no-resolution-filter`, `--no-rmsd-filter`

---

## Output Format

| Column | Description |
|--------|-------------|
| `global_score`, `pocket_ca_score`, `pocket_all_score` | ConfBench scores |
| `rmsd_pred_holo_*` | RMSD between prediction and holo |
| `rmsd_pred_apo_*` | RMSD between prediction and apo |
| `rmsd_apo_holo_*` | RMSD between apo and holo |
