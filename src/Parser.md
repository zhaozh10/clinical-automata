## Task Interpretation (Doctor → AI)

The agent must first convert the doctor's natural language request into a structured task specification.

This is a **critical step**.

### Required fields:

```json
{
  "task_type": "classification | regression | segmentation | detection | multimodal | ...",
  "clinical objective": "...",
  "risk preference": "e.g. prioritize sensitivity, no specific constraints, ...",
  "input_modality": "image | text | tabular | multimodal",
  "expected output format during inference":"prob distribution | prob distribution with explainable heatmap | bbox | seg mask | pure language description | ... ",
  "metrics_summary": ["auc", "f1", "sensitivity", "IoU","..."],
  "primary_metric": "auc",
  "metric_direction": {
    "auc": "higher",
    "f1": "higher"
  },
}
```

### Rules:
- Please first determine the role of the doctor-requested model based on the task type: treat classification tasks as decision-making, and treat detection, segmentation, image-text retrieval, and similar supportive vision tasks as assistive. In many cases, doctors prefer assistive models to decision-making models. Therefore, for decision-making tasks, you should adopt a highly cautious stance, make conservative judgments, and avoid overconfident conclusions.
- **Important!!!**. When only part of the dataset has full supervision for the main task, but additional samples have weaker or coarser labels related to the same clinical target, do not default to training only on the fully labeled subset. Explicitly consider whether the weaker labels can be used.
- The agent must infer missing but clinically important metrics if not specified.
- The selection of primary metric should be consistent with risk preference.
- For medical classification tasks, sensitivity, specificity, and AUC are frequently considered metrics.
- In a doctor-driven setting, overall averaged metric is often not the main target. What matters is whether the model performs well on the few categories that are clinically costly to miss or confuse. Class-specific metrics plus pairwise confusion analysis is the most useful evaluation
- The **primary_metric MUST be defined**.
- Metric direction MUST be explicitly defined (`higher` or `lower`).
