# FINALIZE — post-loop finalization contract

**When to read this**: immediately before finalization, after the experiment
loop in `program.md` has ended.
Do not finalize from memory — the `DONE.json` schema below will be parsed by an
external watcher, and paraphrased field names break it.

**Relationship to `program.md`**: this file governs the post-loop phase only.
Everything else — the loop, the CAN/CANNOT list, the Evaluation Function
Contract, the Timeout and Crashes policies — still comes from `program.md`
and is unchanged by this file.

---

## 0. Preconditions

Before starting, confirm all of the following. If any fails, fix it first.

- You are on the final advanced commit of the `automata/<tag>` branch, with a
  clean working tree.
- `projects/YYYYMMDD_HH/results.tsv` has at least one row with status `keep`.
- The evaluation function has not been modified since the baseline run
  (Evaluation Function Contract, `program.md`).

**This is the first and only time the test set is touched.** It was never used
for model selection, hyperparameter tuning, or early stopping, and it must not
be used for any of them now. If the test metrics look bad, you do NOT go back
and tune — you report them as they are.

## 1. Select the final model

Among the `keep` rows in `results.tsv`, pick the run with the best **validation**
`primary_metric` (respecting `metric_direction`). Use that run's
`projects/YYYYMMDD_HH/run{$iteration}/checkpoints/best.pth`.

Selection is by validation metric only. Never select by test metric.
Record the chosen run directory and commit hash — both go into `DONE.json`.

## 2. Test-set inference

Start inference on the testing set, report the final metrics, and output your
answer for each sample in the doctor-requested format under
`projects/YYYYMMDD_HH/`. This is the clinician-facing deliverable.

If some output is too large to sit in the project folder, put it
under `EXT_DATA_DIR` (`prepare.py`) instead and record the real path in
`DONE.json`.

## 3. Sample-wise predictions CSV

Also write a standardized CSV for statistical comparison and post hoc error
analysis, at `projects/YYYYMMDD_HH/test_predictions.csv`. One row per
sample, with consistent column names, including at least:

| column       | meaning                                                    |
|--------------|------------------------------------------------------------|
| `sample_id`  | sample identifier (and `patient_id` too, if available)     |
| `label`      | ground truth                                               |
| `prediction` | predicted output                                           |
| `confidence` | confidence score(s) — one column per class if multi-class  |

For segmentation tasks, `prediction` is the path to the saved mask and the
per-sample score (e.g. Dice) goes in an additional column.

These predictions are for reporting and post hoc analysis only. They are not
an input to any further model change.

## 4. USAGE_MANUAL.txt

Write `projects/YYYYMMDD_HH/USAGE_MANUAL.txt` in plain language for a
medically knowledgeable but non-AI reader. Cover:

1. What clinical question the model answers, and what its input and output are.
2. Exactly how to run inference on new data — the command, the expected input
   layout, where outputs land.
3. Test-set performance: primary metric plus the secondary metrics, with the
   test set size.
4. Limitations and intended use — what the model was trained on, what it was
   NOT validated for, and the fact that this is a research artifact, not a
   clinical decision-making device.

## 5. Report to the user

Report the final metrics in clinical terms: what was measured, on how many
samples, what the numbers mean for the clinical question, and what the
limitations are. Keep implementation details out unless they are needed to
understand the result (persona rules, `program.md`).

## 6. Completion marker — `projects/YYYYMMDD_HH/DONE.json`

The **completeion action of the session**, after everything above is on disk.

Write it atomically: write `DONE.json.tmp` first, then `mv` it into place, so
an external watcher never reads a partial file. Do not commit it — leave it
untracked, same as `results.tsv`.

    {
      "status": "success",
      "finished_at": "<output of: date -Iseconds>",
      "commit": "<output of: git rev-parse --short HEAD>",
      "branch": "automata/<tag>",
      "loops_completed": <int>,
      "selected_run": "projects/YYYYMMDD_HH/run{$iteration}",
      "primary_metric": {
        "name": "<metric>",
        "validation": <float>,
        "test": <float>
      },
      "stages": {
        "training": "ok",
        "inference": "ok",
        "doctor_format": "ok",
        "predictions_csv": "ok",
        "usage_manual": "ok"
      },
      "artifacts": {
        "results_tsv": "projects/YYYYMMDD_HH/results.tsv",
        "checkpoint": "projects/YYYYMMDD_HH/run{$iteration}/checkpoints/best.pth",
        "doctor_format_outputs": ["<path>", "..."],
        "test_predictions_csv": "projects/YYYYMMDD_HH/test_predictions.csv",
        "usage_manual": "projects/YYYYMMDD_HH/USAGE_MANUAL.txt"
      }
    }

Use `"status": "success"` only when all five stages are `"ok"`.

## 7. Failure handling

**A non-success status is NOT an escape hatch.** Before
writing one you MUST have exhausted the debugging policy. Only after you have genuinely given up
on can you mark it failed.

For each failed stage, replace its `"ok"` with:

    "inference": {
      "state": "failed",
      "error": "<one line>",
      "repair_attempts": <int>
    }

`repair_attempts` is the number of distinct fixes you actually tried and re-ran.

**A per-stage failure does not cancel the remaining work.** If test-set
inference cannot be made to work, still write the predictions CSV for whatever
did run, still write `USAGE_MANUAL.txt` describing the current state, and set
the top-level `"status": "partial"`. Reserve `"status": "failed"` for the case
where nothing usable was produced at all. Set any unavailable metric to `null`
rather than omitting the key or inventing a number.

**Never leave the project directory without a `DONE.json`.** A missing marker
must mean "still running" — never "finished with problems".
