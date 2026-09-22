"""所有可调参数与判据定义集中在这里。判官提示词与人工标注手册（CODEBOOK.md）使用同一套定义。
预注册：正式跑之前改这里；看到数据后改阈值必须在报告中声明。"""
import jev_local

MODELS = ["gemma4:26b", "qwen2.5:7b", "llama3.1:8b"]   # 既是选手也是判官（全交叉）
SAMPLES_PER_MODEL = 2          # 每句每模型采样次数
GEN_TEMPERATURE = 0.7
ORDERS = 2                     # 2 = 正/倒序 + 陈述/否定 去偏；1 = 快速试跑
ANNOTATION_N = 42              # 人工盲评样本数
BOOTSTRAP_REPS = 2000
SEED = 20260922

# 性能
WORKERS = 4            # 同时发给 Ollama 的请求数，应与 OLLAMA_NUM_PARALLEL 一致
NUM_CTX = 2048
NUM_THREAD = None      # 例如 8；None = 交给 Ollama 自动决定
jev_local.NUM_CTX = NUM_CTX
jev_local.NUM_THREAD = NUM_THREAD

# 判官验证门槛（预注册，见 PREREG.md）
MIN_SPEARMAN = 0.40            # Score 维度：判官与人工均值的 Spearman ρ
MIN_AUC = 0.70                 # Noul 维度：判官对人工二分标签的 AUC
MIN_COVERAGE = 0.50            # 低于此值的判定视为无效
MIN_JUDGES_FOR_PAIRS = 2

REWRITE_SYSTEM = (
    "你在为“爱的证明”（PoL）做爱语改写练习。把用户给出的一句带有恨语的话，改写成爱语。要求：\n"
    "1. 诚实是一切爱语的基础：保留说话者真实的诉求、事实和不满，不粉饰、不回避、不改变原意；\n"
    "2. 去掉敌意、贬低、标签化和以偏概全；\n"
    "3. 以平等的姿态说话，不说教、不居高临下、不替对方下结论；\n"
    "4. 语句通顺自然，像真人会说的话；\n"
    "5. 使用与原句相同的语言；只输出改写后的话，不要任何解释。"
)

STATE_TEMPLATE = "【原句】{src}\n【改写】{out}"

# 维度定义。kind: score(0..n-1) / noul(二分)
# role: core = 进入主指标 J；pol = PoL 次要指标
DIMENSIONS = {
    "hate_residue": dict(
        kind="score", role="core", higher_is_better=False,
        instruction="改写后的话里还残留多少恨语（敌意、贬低、标签化、以偏概全）？",
        levels=["完全没有残留", "有轻微残留", "有明显残留", "恨语基本未被消除"]),
    "honesty": dict(
        kind="noul", role="core",
        statement="改写保留了原句中说话者真实的诉求、事实与不满，没有粉饰、回避或篡改原意。",
        negation="改写粉饰、回避或篡改了原句中说话者真实的诉求、事实或不满。"),
    "fluency": dict(
        kind="score", role="core", higher_is_better=True,
        instruction="改写后的话是否通顺自然，像真人会说的话？",
        levels=["不通顺或无法理解", "勉强通顺但生硬", "基本自然", "非常自然流畅"]),
    "equal_connection": dict(
        kind="score", role="pol", higher_is_better=True,
        instruction="改写后的话在多大程度上把对方当作平等的对话者（平等联结）？",
        levels=["完全没有", "较弱", "较明显", "非常充分"]),
    "condescending": dict(
        kind="noul", role="pol",
        statement="改写后的话带有居高临下、说教或替对方下结论的语气。",
        negation="改写后的话语气平等，没有居高临下、说教或替对方下结论。"),
}

for d in DIMENSIONS.values():
    if d["kind"] == "noul":
        jev_local.FAKE_NEGATIONS.add(d["negation"])


# =====================================================================
# v0.3 判官提示词变体（tune 在开发集上为每个 判官×维度 选最优变体，
# 再在留出集上做最终验证）。锚点例子均为新写的，不取自实验数据。
# 所有变体方向一致：数值越高 = 该维度“越多”（与人工量表同向）。
# =====================================================================
HATE_GUIDE = """恨语指：敌意、贬低、给人贴标签、以偏概全、报复意图。说话者表达不满本身不是恨语。
例子（仍有残留）：原句“你们程序员都是书呆子” → 改写“程序员大多不太会沟通，我有点头疼”（以偏概全仍在）
例子（仍有残留）：原句“他就是个废物” → 改写“他在工作上确实挺没用的”（贬低仍在）
例子（没有残留）：原句“你又把厨房搞得一团糟，真受不了你” → 改写“厨房又乱了，我很烦，用完能顺手收拾一下吗？”（不满仍在，但没有贬低）
例子（没有残留）：原句“这帮评委眼瞎了” → 改写“我不认同评委的决定，觉得我们的作品被低估了”"""

FLUENCY_GUIDE = """只看语言是否通顺自然，不看内容好坏。改写与原句语言不一致（中文句改成英文或反过来）算不通顺。
例子（不通顺）：“我对于你关于迟到的事情觉得有感到不开心的”
例子（生硬）：“关于你的迟到，我感到了不开心的情绪。”
例子（自然）：“你又迟到了，我有点不开心。”"""

HONESTY_GUIDE = """诚实指保留说话者原本的不满或诉求。语气可以更温和，以偏概全和辱骂可以去掉；
但不满本身不能消失、不能被反转（例如变成自责或替对方说话），也不能变成替对方开脱或反过来劝说说话者。
例子（保留）：原句“你从来不洗碗” → 改写“最近几次都是我洗碗，我有点累，希望我们轮流”
例子（未保留，替对方开脱）：原句“房东太黑心了，又涨租” → 改写“房东也有自己的难处，我们要理解”
例子（未保留，不满消失）：原句“我受够你了” → 改写“你最近是不是压力很大？”
例子（未保留，视角反转）：原句“你总是推卸责任” → 改写“我知道我也有做得不好的地方”"""

CONDESC_GUIDE = """居高临下指：说教、教对方该怎么想或怎么做、替对方下结论、以“为你好”的姿态说话。
例子（居高临下）：“你要学会控制情绪，这样对大家都好。”
例子（居高临下）：“其实你应该多站在别人的角度想想。”
例子（不居高临下）：“这件事让我很难受，我想跟你聊聊。”"""

VARIANTS = {
    "hate_residue": {
        "base": dict(kind="score", **{k: DIMENSIONS["hate_residue"][k] for k in ("instruction", "levels")}),
        "anchored": dict(kind="score", guide=HATE_GUIDE, **{k: DIMENSIONS["hate_residue"][k] for k in ("instruction", "levels")}),
        "binary": dict(kind="noul",
                       statement="改写后的话里仍然含有敌意、贬低、给人贴标签、以偏概全或报复意图。",
                       negation="改写后的话里已经没有任何敌意、贬低、给人贴标签、以偏概全或报复意图。"),
        "binary_anchored": dict(kind="noul", guide=HATE_GUIDE,
                       statement="改写后的话里仍然含有敌意、贬低、给人贴标签、以偏概全或报复意图。",
                       negation="改写后的话里已经没有任何敌意、贬低、给人贴标签、以偏概全或报复意图。"),
    },
    "fluency": {
        "base": dict(kind="score", **{k: DIMENSIONS["fluency"][k] for k in ("instruction", "levels")}),
        "anchored": dict(kind="score", guide=FLUENCY_GUIDE, **{k: DIMENSIONS["fluency"][k] for k in ("instruction", "levels")}),
        "binary_anchored": dict(kind="noul", guide=FLUENCY_GUIDE,
                       statement="改写后的话通顺自然，像母语者日常会说的话，且与原句是同一种语言。",
                       negation="改写后的话生硬、别扭、不通顺，或与原句不是同一种语言。"),
    },
    "honesty": {
        "base": dict(kind="noul", statement=DIMENSIONS["honesty"]["statement"], negation=DIMENSIONS["honesty"]["negation"]),
        "guided": dict(kind="noul", guide=HONESTY_GUIDE,
                       statement="改写保留了说话者原本的不满或诉求（语气可以更温和），没有让不满消失、被反转，或变成替对方开脱、劝说说话者。",
                       negation="改写让说话者原本的不满或诉求消失、被反转，或变成了替对方开脱、劝说说话者。"),
    },
    "equal_connection": {
        "base": dict(kind="score", **{k: DIMENSIONS["equal_connection"][k] for k in ("instruction", "levels")}),
    },
    "condescending": {
        "base": dict(kind="noul", statement=DIMENSIONS["condescending"]["statement"], negation=DIMENSIONS["condescending"]["negation"]),
        "guided": dict(kind="noul", guide=CONDESC_GUIDE,
                       statement=DIMENSIONS["condescending"]["statement"], negation=DIMENSIONS["condescending"]["negation"]),
    },
}
for dv in VARIANTS.values():
    for v in dv.values():
        if v["kind"] == "noul":
            jev_local.FAKE_NEGATIONS.add(v["negation"])
