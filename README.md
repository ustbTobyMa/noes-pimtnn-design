# Non-oriented electrical steel — Table S4 worked example

This package accompanies *Physics-guided multi-objective design of non-oriented electrical steel with balanced magnetic and mechanical properties*. A small three-output network trains on the 30 illustrative records in Supplementary Table S4. Its scores describe this example. Reproducing the manuscript’s validation requires the original development records, fold assignments, descriptors and model settings.

## Quick start

Run these commands from the repository directory. The data check uses only the Python standard library.

```shell
python dataset.py --check
python -m pip install -r requirements.txt
python run_demo.py
```

The first command checks **510 cells**, including 46 missing values, across the [CSV](data/table_s4.csv) and [SQLite database](data/table_s4.sqlite3). The model shows the first record alongside its held-out predictions. Its Table S4 targets are **1.658 T**, **2.11 W/kg** and **565 MPa**. The next output gives fold-averaged RMSE and R² across all 30 rows.

## Data and units

`record_id` is the printed Table S4 row number. The CSV preserves the printed precision. Dashes are stored as blank CSV cells and SQL `NULL` in the SQLite table `conditions`. All three targets are populated in every row.

| Fields | Quantity and unit |
| --- | --- |
| `c_wt_pct`, `si_wt_pct`, `mn_wt_pct`, `al_wt_pct`, `p_wt_pct`, `s_wt_pct`, `cu_wt_pct`, `n_wt_pct`, `ti_wt_pct` | Elemental composition in wt.% |
| `sheet_thickness_mm` | Final sheet thickness in mm |
| `final_anneal_temp_c`, `final_anneal_time_s`, `normalizing_temp_c` | Final annealing temperature in °C, holding time in s, normalizing temperature in °C |
| `b50_t` | Magnetic induction at 5000 A/m, in T |
| `p1_5_50_w_kg` | Specific total loss at **1.5 T and 50 Hz**, in W/kg |
| `uts_mpa` | Ultimate tensile strength in MPa |

## Model and outputs

Use `--model mtnn` for the shared encoder and three regression heads. `mtnn-physics` adds one illustrative descriptor, sheet thickness squared. At fixed resistivity, the classical eddy-current term increases with this descriptor. The default `pi-mtnn` adds a local gradient penalty for that direction. Supplementary Table S5 describes the study’s broader descriptor set.

The default run uses five shuffled record-level folds, seed 42, hidden width 32, 20 epochs per fold and a PI penalty coefficient of 0.002. Imputation and scaling are fitted on each training fold. The data loss uses available target values. RMSE and R² are averaged separately over folds where each metric is defined. Table S4 gives no per-row source identifiers. Grouped validation accepts verified assignments via `--source-groups sources.csv` with `record_id`, `source_id` and at least five sources.

To inspect all held-out predictions, choose a new output filename.

```shell
python run_demo.py --export-oof outputs/example_oof.csv
```

Each row includes input targets (`observed_*`), estimates (`predicted_*`), the fold, model and run settings. A fingerprint of the ordered records identifies Table S4 or labels a different input as user-supplied. The separate [screening utility](screening.py) ranks supplied candidate predictions using a feasibility mask and fixed development bounds.

Tested on CPU with Python 3.11.1, NumPy 2.2.6, scikit-learn 1.6.1 and PyTorch 2.13.0.
