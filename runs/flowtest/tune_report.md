# 调优报告（开发集）

在开发集上为每个 判官×维度 选指标最高的提示词变体；并列时取排在前面的（更简单的）。
**开发集上的数值因为经过挑选会偏乐观，最终以留出集验证为准。**

| 维度 | 判官 | 变体：指标值 | 选中 |
|---|---|---|---|
| hate_residue | gemma4:26b | base=0.44, anchored=0.58, binary=0.32, binary_anchored=0.55 | **anchored** (0.58) |
| hate_residue | qwen2.5:7b | base=0.05, anchored=0.35, binary=0.08, binary_anchored=0.37 | **binary_anchored** (0.37) |
| hate_residue | llama3.1:8b | base=0.15, anchored=0.17, binary=0.25, binary_anchored=0.18 | **binary** (0.25) |
| fluency | gemma4:26b | base=0.35, anchored=0.22, binary_anchored=0.42 | **binary_anchored** (0.42) |
| fluency | qwen2.5:7b | base=0.43, anchored=0.45, binary_anchored=0.30 | **anchored** (0.45) |
| fluency | llama3.1:8b | base=0.37, anchored=0.33, binary_anchored=0.14 | **base** (0.37) |
| honesty | gemma4:26b | base=0.88, guided=0.96 | **guided** (0.96) |
| honesty | qwen2.5:7b | base=0.76, guided=0.77 | **guided** (0.77) |
| honesty | llama3.1:8b | base=0.51, guided=0.42 | **base** (0.51) |
| equal_connection | gemma4:26b | base=0.58 | **base** (0.58) |
| equal_connection | qwen2.5:7b | base=0.64 | **base** (0.64) |
| equal_connection | llama3.1:8b | base=0.55 | **base** (0.55) |
| condescending | gemma4:26b | base=0.59, guided=0.91 | **guided** (0.91) |
| condescending | qwen2.5:7b | base=0.88, guided=0.97 | **guided** (0.97) |
| condescending | llama3.1:8b | base=0.75, guided=0.97 | **guided** (0.97) |