# 爱语擂台 · Love-Language Arena

用三个本地模型（Ollama）把带恨语的句子改写成爱语，再让它们互相评审，检验**本地 LLM 能否在 PoL（爱的证明）的爱语判据上替代人工判断**。

*Three local LLMs rewrite hostile sentences into "love language" (PoL framework) and judge each other. The pipeline tests whether local LLM judges can stand in for human judgment, with full-crossed judging, self-preference correction, per-judge calibration, dev/holdout prompt tuning and a pre-registered negative-result path.*

**当前状态**：工作流已验证。人工金标准尚未采集，现有“判官通过验证”的结论基于 Claude 标注，只能说明流程可用。完整记录见 [`docs/EXPERIMENT_REPORT.md`](docs/EXPERIMENT_REPORT.md)。

## 主要发现

- 改写提示词明确要求“不能粉饰”，但 26–31% 的改写仍把说话者的不满改没了、反转了，或者变成替对方开脱。
- gemma4:26b 是唯一在所有维度上都通过验证的判官；llama3.1:8b 不适合做中文判官；qwen2.5:7b 在量表题上有位置偏差。
- 提示词调优在 42 条的开发集上挑中了噪声：qwen 的诚实判定在开发集上只高 0.01，留出集上 AUC 从 0.79 跌到 0.50。
- 模型排名 gemma > qwen > llama 目前**不可信**：两个核心维度只有 gemma 一个判官，自偏好无法校正。

## 目录

| 路径 | 内容 |
|---|---|
| `run.py` | 主程序，分阶段子命令 |
| `jev_local.py` | 在 Ollama 上用 logprobs 实现类 Jev 的 Choice / Score / Noul，带去位置偏差 |
| `stats_utils.py` | Spearman、AUC、Krippendorff α、Platt 校准、判官效应回归 |
| `config.py` | 模型、判据定义、提示词变体、门槛、性能参数 |
| `data/sentences.txt` | 60 句待改写句子（中文 50，英文 10） |
| `docs/PREREG.md` | 预注册与修订记录 |
| `docs/CODEBOOK.md` | 标注手册 |
| `docs/EXPERIMENT_REPORT.md` | 完整实验记录 |
| `docs/REFERENCES.md` | 参考文献 |
| `tools/` | 诊断脚本：`diagnose.py`、`examples.py`、`fill_dummy.py`（仅测试用） |
| `runs/flowtest/` | 本次全部数据：改写、判定、调优、Claude 标注、报告 |
| `runs/pilot/` | 5 句试跑 |

## 运行（Windows PowerShell）

依赖：Python 3.10+、`numpy`、Ollama ≥ 0.12.11，以及拉取好的三个模型。

```powershell
python -m pip install numpy
ollama pull gemma4:26b; ollama pull qwen2.5:7b; ollama pull llama3.1:8b

python run.py check                              # 检查三个模型能否按格式作答
python run.py generate                           # 360 条改写
python run.py annotate                           # 开发集标注表 → runs\main\annotations\
python run.py judge                              # 全交叉判定（可中断续跑）
python run.py annotate --set holdout             # 留出集标注表 → runs\main\holdout\
# —— 人工标注两张表（见 docs/CODEBOOK.md）——
python run.py tune                               # 开发集上为每个判官×维度选提示词变体
python run.py judge    --tuned
python run.py validate --tuned --set holdout     # 最终准入只看留出集
python run.py analyze  --tuned
python run.py pairs    --tuned                   # 核心维度都有 ≥2 个判官通过才产出偏好对
```

加速建议：设置 `OLLAMA_NUM_PARALLEL=4`、`OLLAMA_FLASH_ATTENTION=1`、`OLLAMA_KV_CACHE_TYPE=q8_0`，然后重启 Ollama；`config.py` 中的 `WORKERS` 与 `OLLAMA_NUM_PARALLEL` 保持一致。

不需要 Ollama 也能复现本次分析：

```powershell
python run.py validate --run flowtest --tuned --set holdout
python run.py analyze  --run flowtest --tuned --quick
```

## 说明

- 仓库里的代码比产生 `runs/flowtest` 结果时多了并发请求，判定逻辑没有变化；在同一份判定数据上重跑分析，点估计完全一致。
- 所有标注文件的备注列都标有“CLAUDE标注(非人工)”。
