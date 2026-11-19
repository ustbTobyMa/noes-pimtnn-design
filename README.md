# Physics-guided multi-objective design of non-oriented electrical steel

This companion to the IJMMM manuscript *Physics-guided multi-objective design of non-oriented electrical steel with balanced magnetic and mechanical properties* provides a runnable model framework and the 30 illustrative records in Supplementary Table S4. The three targets are magnetic induction at 5000 A/m, specific total loss at **1.5 T and 50 Hz**, and ultimate tensile strength.

## Example database

[Download the CSV](data/table_s4.csv) or [open the SQLite database](data/table_s4.sqlite3). Both editions contain the same 30 rows in Table S4 order. `record_id` is the printed row number. Blanks in the CSV and SQL `NULL` values correspond to dashes in the table. The printed numerical precision is retained in the CSV.

| Columns | Meaning and units |
| --- | --- |
| `c_wt_pct`, `si_wt_pct`, `mn_wt_pct`, `al_wt_pct`, `p_wt_pct`, `s_wt_pct`, `cu_wt_pct`, `n_wt_pct`, `ti_wt_pct` | Elemental composition in wt.% |
| `sheet_thickness_mm` | Final sheet thickness in mm |
| `final_anneal_temp_c`, `final_anneal_time_s` | Final annealing temperature in °C and holding time in s |
| `normalizing_temp_c` | Normalizing temperature in °C |
| `b50_t` | Magnetic induction at 5000 A/m, in T |
| `p1_5_50_w_kg` | Specific total loss at 1.5 T and 50 Hz, in W/kg |
| `uts_mpa` | Ultimate tensile strength in MPa |

Inspect the data with Python 3.10 or later. This step uses only the standard library.

```shell
python dataset.py
```

The SQLite table is named `conditions`. For example, the following query returns the three targets for the first five records.

```sql
SELECT record_id, b50_t, p1_5_50_w_kg, uts_mpa
FROM conditions
ORDER BY record_id
LIMIT 5;
```

## Run the model framework

From the repository directory, install the numerical dependencies and run a CPU demonstration.

```shell
python -m pip install -r requirements.txt
python run_demo.py --model pi-mtnn
```

| Model option | Inputs and loss |
| --- | --- |
| `mtnn` | Composition and process variables, shared encoder, three regression heads |
| `mtnn-physics` | Adds a descriptor branch to the same three-target model |
| `pi-mtnn` | Adds a hinge penalty for gradients that oppose a specified local physical direction |

The example descriptor is sheet thickness squared. It illustrates the increasing thickness contribution to classical eddy-current loss at fixed resistivity. Table S4 supplies the thickness. With externally calculated descriptors, pass their row-aligned matrix and the corresponding task-by-descriptor signs to `cross_validate` in `run_demo.py`. Use +1 for a locally increasing response, −1 for a decreasing response and 0 for an unspecified direction. Supplementary Table S5 gives the physical interpretation of the full descriptor set.

The example uses five record-random folds with seed 42. Each fold fits its input imputation, input scaling and target scaling on the training rows. The three heads use equally weighted, per-task errors over available labels. Missing descriptor observations are excluded from the directional penalty. The demonstration uses a hidden width of 32, 20 epochs and a penalty coefficient of 0.002 for `pi-mtnn`.

For each held-out fold, the runner calculates RMSE in physical units and R². It reports the arithmetic mean of each metric across folds for which that metric is defined. To inspect the individual predictions, use a new output path.

```shell
python run_demo.py --model pi-mtnn --export-oof outputs/example_oof.csv
```

Table S4 supplies no per-row source identifiers. For a dataset with verified source assignments, use `--data records.csv --source-groups sources.csv`. The second CSV maps each `record_id` to a `source_id`. The grouped run holds whole sources out and requires at least five distinct sources.

## Screen candidate predictions

`rank_candidates` in [screening.py](screening.py) accepts candidate IDs, the three predicted objectives in database order, a Boolean feasibility mask and fixed lower and upper reference bounds. It maximizes magnetic induction and strength, minimizes loss, and retains feasible members of the original non-dominated set. It returns their IDs and equal-weight normalized distances to the ideal point, ordered by increasing distance. Use the same development bounds for every candidate in a comparison.

The released records illustrate the public data schema and exercise the code. Metrics printed by the example describe only this 30-row run. Reproducing manuscript validation or a particular candidate ranking requires the corresponding development records, fold assignments, model inputs and candidate predictions.
