# Non-oriented electrical steel database and code

Database and code accompanying *Physics-guided multi-objective design of non-oriented electrical steel with balanced magnetic and mechanical properties*.

## Database

The condition records from Supplementary Table S4 are available in two formats. In both, `record_id` is the row number in Table S4. Blank CSV fields correspond to `NULL` in the SQLite table `conditions`. [CSV](data/table_s4.csv) · [SQLite](data/table_s4.sqlite3)

| Fields | Quantity and unit |
| --- | --- |
| `c_wt_pct`, `si_wt_pct`, `mn_wt_pct`, `al_wt_pct`, `p_wt_pct`, `s_wt_pct`, `cu_wt_pct`, `n_wt_pct`, `ti_wt_pct` | Elemental composition in wt.% |
| `sheet_thickness_mm` | Final sheet thickness in mm |
| `final_anneal_temp_c`, `final_anneal_time_s`, `normalizing_temp_c` | Final annealing temperature in °C, holding time in s, normalizing temperature in °C |
| `b50_t` | Magnetic induction at 5000 A/m, in T |
| `p1_5_50_w_kg` | Specific total loss at 1.5 T and 50 Hz, in W/kg |
| `uts_mpa` | Ultimate tensile strength in MPa |

## Running the analysis

```shell
python dataset.py --check
python -m pip install -r requirements.txt
python cross_validate.py
```

The first command checks that the CSV and SQLite records agree. The last command trains PI-MTNN on the Table S4 records using five shuffled folds and reports mean fold RMSE and R² for each property. Imputation and scaling are fitted on the training records in each fold. To save the held-out predictions for each record, run

```shell
python cross_validate.py --export-oof outputs/predictions.csv
```

Select a model with `--model`. The `mtnn` model uses a shared encoder and three prediction heads. The `mtnn-physics` model adds squared sheet thickness as a descriptor. The `pi-mtnn` model also penalizes negative gradients of predicted specific total loss with respect to that descriptor.

The default settings use seed 42 and 20 training epochs per fold. To group the folds by literature source, pass `--source-groups sources.csv` with columns `record_id` and `source_id` and at least five distinct sources.

For candidate screening, `rank_candidates` takes predicted properties, feasibility flags and fixed normalization bounds. It ranks feasible members of the Pareto front by their distance to the ideal point. [Screening code](screening.py)
