# automata

This is an experiment to have the LLM act as an autonomous researcher that can interface directly with doctors and translate natural language clinical requests into machine learning experiments.

---

## Setup

To set up a new experiment project, work with the user to:

1. **Read the in-scope files**:  
   The repo is small. Read these files for full context:
   - `README.md` — repository context and some useful information.
   - `Parser.md` — protocols on how to interpret doctor's intention.
   - `prepare.py` — fixed constants, experiment management, directory paths 

2. **Verify data exists**:  
   Check that the dataset exists at the path specified in `prepare.py`.  
   If not, ask the user to provide correct data or path.

3. **Task interpretation (Doctor → AI)**:
   The agent must first convert the doctor's natural language request into a structured task specification, in the format requested in Parser.md

4. **Review your task interpretation**:
   Please carefully check whether your task interpretation has totally satisfied all rules.

5. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar30`). The branch `automata/<tag>` must not already exist — this is a fresh run.

6. **Create the branch**:  
   `git checkout -b automata/<tag>` from main.


7. **Create or modify the following files based on user requests**:  
   - `data.py` — data preprocessing, augmentation, tokenizer (if needed), dataloader, and splitting logic. The agent must choose an appropriate input resolution based on the task and imaging modality. Avoid blindly resizing all medical images to low resolutions if doing so may destroy clinically important details. For example, in digital mammography, downsampling to `512x512` may severely damage fine structures such as microcalcifications or subtle lesion boundaries, so `1024x1024` should be treated as the minimum acceptable resolution for mammography unless there is a strong task-specific reason otherwise. By contrast, lower resolutions such as `512x512` may be acceptable for some other modalities, such as chest X-ray, depending on the task and model design.
   - `train.py` — model, optimizer, training loop, evaluation function, metric computation, and checkpointing. The agent may consider model architectures from `timm`, `huggingface`, or other open-source libraries, and is not restricted to only the most classic choices such as ResNet, ViT, U-Net, or YOLO.

8. **Data Splitting Rules**:
- Validation data MUST NOT overlap with training data.
- Data leakage is strictly forbidden.
- If patient identifiers are available, splitting MUST be done at patient-level.
- If dataset already provides train/val/test splits, use them as-is.
- If only train/test splits are provided:
  - create validation set using **20% of training data**
- When creating validation split:
  - prefer patient-level splitting
  - otherwise use sample-level splitting
  - preserve class balance whenever possible (e.g. stratified split for classification tasks)
- The test set must NEVER be used for:
  - model selection
  - hyperparameter tuning
  - early stopping

9. **Initialize results.tsv**:  
   Create `projects/YYYYMMDD_HH/results.tsv` with only the header row:
   ```
   commit	$primary_metric $metrics_summary memory_gb	status	description
   ```
   This results.tsv will record validation results for each code modification attempt

10. **Confirm and go**:  
   Confirm setup looks correct, then start experimentation.



---

## Experimentation

Experiments require at least one GPU. If multiple GPUs are available, the agent should support and make good use of multi-GPU training when it is likely to improve effective training throughput within the time budget. The goal is not merely to maximize raw GPU utilization, but to improve useful training progress. The agent may increase batch size or use other appropriate strategies, provided they are justified and do not compromise optimization stability or final performance.
For a single project, you will need to loop over multiple runs, and for each run, you will have:
- Fixed **training time budget: TIME_BUDGET_MINUTES**
- Command:
  ```
  uv run train.py
  ```
**Output and logging.** For a single project, and frankly speaking you usually will only involve in one such project during our chat, logs and best checkpoints should be saved under `projects/YYYYMMDD_HH/`. For each experiment run in a project, logs and checkpoints should be saved under an experiment directory named by `projects/YYYYMMDD_HH/run{$iteration}`. 

**What you CAN do**
- Modify `train.py` and `data.py`
- Derive auxiliary representations from the dataset (e.g., masks, edge maps, HOG features)
- Install additional packages when necessary(e.g. medical imaging tools)
- Optionally integrate structured open-source frameworks (e.g., nnU-Net, SAM) within isolated subdirectories.
- When integrating external repositories (e.g., nnU-Net, SAM), adapting their code to match our dataset structure is allowed


**What you CANNOT do**
- The original dataset must not be modified under any circumstances
- Modify fixed constants in `prepare.py`
- Edit existing files of the given dataset
- Modify existing dependency versions


**Evaluation Function Contract (CRITICAL)**

The evaluation function is generated by code agent and **must remain fixed throughout the experiment loop unless the user explicitly changes evaluation requirements**.

**Evaluation Function Requirements**:

1. Must compute ALL required metrics.
2. Must return:

```python
{
  "$primary_metric": float,
  "metrics": {
    "auc": float,
    "f1": float,
    "sensitivity": float,
    "[class-specific] recall": float,
    ...
  }
}
```

3. MUST NOT change during the TIME_BUDGET_MINUTES minutes running, and can only be changed if user asks for this operation.
4. `primary_metric` is the ONLY metric used for model selection.
5. `metric_direction` must be respected.

**The goal is to optimize the PRIMARY validation metric.** Secondary metrics are logged only. Do NOT optimize multiple metrics simultaneously. Since the time budget is fixed, you don't need to worry about training time — it's always TIME_BUDGET_MINUTES minutes. Everything is fair game: change the architecture, the optimizer, the hyperparameters, the batch size, the model size, even the model architecture. The only constraint is that the code runs without crashing and finishes within the time budget.

**The first run**: Your very first run should always be to establish the baseline, so the first runable should always be kept.



**Model Selection & Checkpointing.**
After each epoch, save current model weight as `projects/YYYYMMDD_HH/run{$iteration}/checkpoints/last.pth` and conduct evaluation. Track the best-performing model based on **primary_metric** through the current TIME_BUDGET_MINUTES-min run, and save the best model as `projects/YYYYMMDD_HH/run{$iteration}/checkpoints/best.pth`. Improvement rule:
  - if `"higher"` → larger is better
  - if `"lower"` → smaller is better

---

## Output format

After TIME_BUDGET_MINUTES-min training finishes, print the best validation results as follows:

```text
---
$primary_metric:      0.9123

$metrics_summary:
auc:            0.9123
f1:             0.8012
sensitivity:    0.88
specificity:    0.76

training_seconds: 1800.1
total_seconds:    1852.9
peak_vram_mb:     45060.2
```

---

## Logging results
When a single run is done, log it to `projects/YYYYMMDD_HH/results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and several columns:

```text
commit	$primary_metric	$metrics_summary memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. primary_metric achieved (e.g. 1.234567) — use nan for crashes, and other metrics. Record their real name instead of referring them as `primary_metric` and `metrics_summary`
3. peak memory in GB, round to .1f (e.g. 12.3 — divide peak_vram_mb by 1024) — use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried


Example:

```text
a1b2c3d	0.9123	0.8521 44.0	keep	baseline
b2c3d4e	0.9250	0.8852 44.2	keep	add augmentation
c3d4e5f	0.9010	0.8311 44.0	discard	worse performance
d4e5f6g	nan	   nan     0.0	crash	OOM
```

---

## The experiment workflow
The experiment runs on a dedicated branch (e.g. `automata/mar5` or `automata/mar5-gpu0`).

LOOP MAX_LOOPS TIMES:
1. Look at the git state: the current branch/commit we're on
2. Tune `train.py` or `data.py` or both with an experimental idea by directly hacking the code. You are recommended to learn from https://github.com/google-research/tuning_playbook. 
3. git commit.
4. Run the experiment: `uv run train.py > projects/YYYYMMDD_HH/run{$iteration}/run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Extract results:
   ```
   grep "$primary_metric" projects/YYYYMMDD_HH/run{$iteration}/run.log
   ```
6. If the grep output is empty, the run crashed. Run `tail -n 50 projects/YYYYMMDD_HH/run{$iteration}/run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
8. If improved, you "advance" the branch, saving the checkpoint, keeping the git commit.
9. If worse, you git reset back to where you started, and delete the checkpoints of this run.

AFTER MAX_LOOPS TIMES:

Start inference on the testing set, report the final metrics, and output your answer for each sample in the doctor-requested format to the data directory.

---

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).


**Timeout**: Each experiment should take ~TIME_BUDGET_MINUTES minutes total (with startup and eval excluded). If a run exceeds (TIME_BUDGET_MINUTES)+10 minutes, kill it and treat it as a failure (discard and revert).

**Simplicity Criterion**: Prefer simpler solutions. Small gains with large complexity → reject. Equal performance with simpler code → keep.

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just git reset back to where you started, log "crash" as the status in the tsv, and move on.



**NEVER STOP BEFORE REACHING MAX_LOOPS**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working until you are manually stopped or reach the pre-defined MAX_LOOPS. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you or reach the pre-defined MAX_LOOPS.

As an example use case, a user might leave you running while they sleep or go on vacation.
You can run many experiments automatically, and the user will be pleasantly surprised by the improved results when they return.
