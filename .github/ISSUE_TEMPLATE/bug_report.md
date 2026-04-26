---
name: Bug report
about: Something is broken in the corpus builder, evaluation pipeline, or results
title: "[BUG] "
labels: bug
assignees: ''
---

**What happened?**

<!-- A clear description of the bug. -->

**Steps to reproduce**

```bash
# Paste the exact command that failed
```

**Expected behaviour**

<!-- What should have happened? -->

**Error output**

```
# Paste the full traceback here
```

**Environment**

```
python -c "import poolbench, torch, datasets, transformers; print(poolbench.__version__, torch.__version__, datasets.__version__, transformers.__version__)"
```

**Hardware** (GPU model, VRAM if relevant)
