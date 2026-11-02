# Ding-format CES-FL / FairForge-2 comparison

- `ding_ff2_comparison.pdf`: compiled report. Tables 1–3 put the single selected CES-FL arm, R2 (partial fine-tuning + DRO), on the last row of each dataset/attribute group and shade it light blue; its values are not bold. Table 3 labels its backbone as Xception beside Ding et al.'s ResNet-50 rows.
- R2 is the highest observed four-dataset mean AUC among archived CES-FL Xception arms (78.68%), selected post hoc and identified as exploratory.
- Table 4 includes available Ding-style FF2 fairness metrics for FF++-trained transfer checkpoints. Fairness values for FF2-trained in-domain score rows and older CFES-D are unavailable in matching saved artifacts; current CES-FL FF2 is marked not run.
- `ding_ff2_comparison.tex`: editable LaTeX source; compile with `pdflatex -interaction=nonstopmode ding_ff2_comparison.tex` twice.
- `source_tables_1_2_3_actual_results.csv`: full-precision CES-FL source comparison rows.

CES-FL evidence snapshot commit: `39471dc2ab13d51d74d0cfb4fe4052109e20b653`. CES-FL rows are single-seed. Ding and CES-FL thresholds/test inventories are not verified to match, so the report does not claim an apples-to-apples win.
