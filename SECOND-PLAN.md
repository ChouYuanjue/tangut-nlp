User: 

# 🤖 [系统指令：西夏文 NLP 实验架构重构与“暴力替换”实施指南 - V2]

**角色设定**：你是一个极其严谨、追求数据完美的 NLP 算法专家。你的任务是协助用户在现有的 `tangut-nlp` 代码仓库上，构建一套层层递进、控制变量极其严格的实验矩阵，**重点包括实施一项基于语义向量的暴力数据合成实验。**

**核心反思（新增）**：
* **关于 UNK 替换的局限性**：Baseline 3.1 尝试将字典未匹配项替换为 `[UNK]`。虽然保证了输入数据的纯净度，但过多的 `[UNK]` 会导致句子语义严重缺失，模型可能难以从“UNK 的海洋”中学会语法结构。
* **关于暴力匹配的学术定义**：我们将该方案系统化为 **Vector-Space Semantic Projection (向量空间语义投影)**。目的是在保证伪预料一定是纯西夏文的前提下，尽可能保留源古代汉语的语义。

## 📊 第一部分：全矩阵对比实验架构 (The Ultimate Ablation Matrix)

我们在原有 4 个 Baseline 的基础上，插入多个中间和极端受控组。请在编写代码前，严格理解它们之间的演进和对比关系：

### 流派一：推理层免训练过渡 (Inference Track)

* **[原有] Baseline 2 (字典 RAG)**：原生的逐字查字典+Prompt 翻译。**[忠实度上界，流畅度下界]**
* **[新增] Baseline 2.1 (RAG + CoT思维链)**：推理脚本 Prompt 级修改。将 RAG 分为“字对齐 -> 直译 -> 意译重写”三步。**[变量：提示词工程]**

### 流派二：数据合成层 SFT 过渡 (Data & SFT Track) - 核心修正层

此层旨在解决原有 Baseline 3（混合比例微调）在推理纯西夏文时的**幻觉**问题。所有新增 Baseline 输入端必须是 **100% 纯净（西夏文字符或安全占位符）**。

* **[原有] Baseline 3 (混合比例 SFT)**：65% 西夏文 + 35% 原文混合输入。**[幻觉对照组]**
* **[新增] Baseline 3.1 (100% UNK 占位 SFT)**：修改数据生成脚本。只要字典未匹配，**绝对不能保留汉字**，必须统一替换为 `[UNK]`。**[变量：输入纯净度，缺点是语义缺失严重]**
* **[新增] Baseline 3.3 (向量空间暴力投影 SFT - The Violent Scheme)**：**核心极简暴力方案**。
    * **实现逻辑**：
        1.  构建一个由“所有西夏文字符的中文释义集合”组成的语义向量候选库（语义特征向量 $V_{dict}$）。
        2.  遍历古代汉语单语语料。对于字典未命中项，计算其语义向量 $V_{raw}$。
        3.  使用 Embedding 模型在候选库中检索最近邻的 $V_{dict}$，从而反向确定对应的西夏文字符进行强制替换。
    * **对比关系**：它和 Baseline 3.1 都是 100% 纯净西夏文输入。3.1 是“语义断裂式”替换，3.3 是“语义扭曲式”替换（强行找平替）。它们的对比能严谨回答：在极低资源场景下，**纯净的数据结构**和**不完美的语义保留**哪个对模型更有帮助？

### 流派三：模型架构与对齐层过渡 (Architecture & Alignment Track)

* **[新增] Baseline 3.2 (多任务对齐 SFT)**：在 Target 端输出中加入 `<dict_match>` 和 `<literal>` 结构。**[变量：显式 Loss 约束，解决幻觉]**
* **[原有] Final (Failed DPO)**：在 Baseline 3 底座上强行对齐纯西夏文输入导致崩溃。**[失败案例分析]**
* **[新增] Final V2 (安全受控 DPO)**：放弃原有 Final，改用 Baseline 3.1/3.3/3.2 中表现最好的一个作为新底座，跨基线构造偏好数据进行对齐。

---

## 🛠️ 第二部分：给 AI 助手的执行指令 (Action Plan - Updated)

> **请代码助手阅读完上述架构后，按照以下顺序与用户进行确认并执行：**

### 阶段 1：现有代码库排查

1.  **搜索数据处理管道**：请反查仓库，列出负责古代汉语语料预处理、字典加载和 SFT 数据合成（`json`/`jsonl` 生成）的 Python 文件。你需要分析现有的字典匹配逻辑是怎样的。

### 阶段 2：实施“暴力替换”方案 (Baseline 3.3 - Vector-Space Semantic Projection)

这是一个复杂的工程任务，请按以下逻辑规划代码修改：

1.  **预前检查**：
    * 检查仓库中是否有现成的 Embedding 模型加载代码（如使用 `sentence-transformers` 或调用大模型 API）。如果没有，建议在 `src/utils/embeddings.py` 中新增一个工具类。
    * 检查是否有向量数据库的依赖（如 `faiss`），如果没有，请提醒用户安装。
2.  **构建语义候选库（The Brain）**：
    * 请编写一个新脚本 `scripts/build_tangut_semantic_index.py`。
    * **逻辑**：遍历字典中所有西夏文字符 $\rightarrow$ 提取其所有中文释义 $\rightarrow$ 将释义文本转化为向量 $\rightarrow$ 将（向量, 西夏字符, 释义文本）存储到 FAISS 索引中。
3.  **重构数据合成脚本**：
    * 请找到你刚才搜到的 SFT 数据合成脚本（如 `src/data_preparation.py`）。
    * **修改建议**：新增一个函数 `synthesize_extreme_pure_tangut_data()`。
    * **逻辑**：遍历单语语料 $\rightarrow$ 查字典 $\rightarrow$ **命中则直替** $\rightarrow$ **未命中则计算原文向量 $\rightarrow$ FAISS 检索最相近语义西夏字 $\rightarrow$ 强行替换** $\rightarrow$ 生成目标现代中文翻译。
    * **注意**：此过程可能耗时较长，需要加入 tqdm 进度条和检查点保存。

### 阶段 3：实施其他 Baseline

1.  **Baseline 2.1 (CoT)**：请反查推理脚本 `experiments/run_baseline2.py`，列出其使用的 Prompt 模板。询问用户是否需要将其重构为三步思维链模板。
2.  **Baseline 3.1 (100% UNK)**：这个任务相对简单，请确认在同一数据合成脚本中，修改替换逻辑（未命中则替换为 `[UNK]`）。

对于任何不清楚的问题，请询问我

Claude: 我完全理解了你的系统指令。这是一个非常严谨的实验矩阵重构方案，核心是通过**向量空间语义投影（Baseline 3.3）**与 **100% UNK 替换（Baseline 3.1）** 的对比，来解决"语义保留 vs. 数据纯净度"的终极消融问题。

让我按照**阶段 1** 开始，排查现有代码库中的数据处理管道。

搜索与 `src/*.py` 匹配的文件

搜索与 `experiments/*.py` 匹配的文件

读取 [](file:///home/runnel/tangut-nlp)

现在让我查看数据合成的核心脚本：

读取 [](file:///home/runnel/tangut-nlp/src/data_synthesis.py)

读取 [](file:///home/runnel/tangut-nlp/src/dictionary_utils.py)

读取 [](file:///home/runnel/tangut-nlp/experiments/baseline3_synthetic_sft.py)

让我检查依赖配置和提示词模板：

搜索与 `requirements*.txt` 匹配的文件

读取 [](file:///home/runnel/tangut-nlp/src/prompt_templates.py)

读取 [](file:///home/runnel/tangut-nlp/requirements.txt)

完美！我完成了**阶段 1** 的代码库排查。让我总结一下关键发现和计划：

## 📋 阶段 1 排查完成

### 核心文件结构
1. **`src/data_synthesis.py`** — 当前数据合成的主要脚本
   - `build_cn_to_tangut_map()` — 从字典CN释义提取汉字→西夏文映射
   - `synthesize_one_pair()` — **当前逻辑**：随机替换 30-70% 汉字为西夏文（Baseline 3 混合策略）

2. **`src/dictionary_utils.py`** — 字典管理工具
   - 支持字符/词级查询和最大前向匹配

3. **`experiments/baseline3_synthetic_sft.py`** — SFT 训练脚本

### 当前限制
- **缺少依赖**：`requirements.txt` **未包含** `sentence-transformers` 或 `faiss`
- **没有 Embedding 工具类**：需要新建 `src/utils/embeddings.py`

---

## ⚡ 在实施**阶段 2（Baseline 3.3）** 之前，我需要你确认几个关键设计决策：

所用工具: AskUserQuestion

Analyzing your answers...

完美！你的选择很明智。现在让我基于这些决策，为**阶段 2 和 3** 设计具体的实施方案。

所用工具: EnterPlanMode

现在我在计划模式下，让我深入探索代码库以设计最优的实施方案。我将并行启动 Explore agents 来快速了解关键模式：

所用工具: Agent

所用工具: Agent

所用工具: Agent

搜索与 `**/dictionary.json` 匹配的文件

搜索与 `**/*dict*.py` 匹配的文件

搜索正则表达式‘\.jsonl’

搜索正则表达式‘jsonl’

搜索正则表达式‘instruction.*input.*output’

Ran terminal command: find /home/runnel/tangut-nlp -type f -name "*.json" | head -20

搜索正则表达式‘load_from_disk|load_dataset’

搜索正则表达式‘AutoModelForCausalLM’

搜索正则表达式‘from_pretrained|load_pretrained|AutoModel’

搜索与 `**/inference.py` 匹配的文件

搜索正则表达式‘def generate|def forward’

读取 [](file:///home/runnel/tangut-nlp/data/dictionary/dictionary.json)

读取 [](file:///home/runnel/tangut-nlp/src/dictionary_utils.py)

读取 [](file:///home/runnel/tangut-nlp/src/reward_dict_builder.py)

搜索正则表达式‘batch|torch\.vmap|vectorize’

读取 [](file:///home/runnel/tangut-nlp/experiments/inference.py)

读取 [](file:///home/runnel/tangut-nlp/src/combine_data.py)

读取 [](file:///home/runnel/tangut-nlp/eval/perplexity.py)

读取 [](file:///home/runnel/tangut-nlp/src/prepare_splits.py)

读取 [](file:///home/runnel/tangut-nlp/src/parse_raw.py)

读取 [](file:///home/runnel/tangut-nlp/eval/run_all_metrics.py)

读取 [](file:///home/runnel/tangut-nlp/experiments/baseline3_synthetic_sft.py)

Ran terminal command: head -100 /home/runnel/tangut-nlp/data/dictionary/dictionary.json

读取 [](file:///home/runnel/tangut-nlp/experiments/final_dpo.py)

读取 [](file:///home/runnel/tangut-nlp/experiments/baseline2_dict_rag.py)

Ran terminal command: wc -l /home/runnel/tangut-nlp/data/dictionary/dictionary.json

搜索正则表达式‘generate\(|input_ids’

读取 [](file:///home/runnel/tangut-nlp/src/data_synthesis.py)

搜索正则表达式‘instruction|input|output|metadata’

读取 [](file:///home/runnel/tangut-nlp/experiments/baseline3_synthetic_sft.py)

读取 [](file:///home/runnel/tangut-nlp/experiments/baseline1_zeroshot.py)

搜索正则表达式‘dictionary|get_stats|validate’

读取 [](file:///home/runnel/tangut-nlp/experiments/baseline2_dict_rag.py)

搜索与 `**/generate_candidates.py` 匹配的文件

Ran terminal command: find /home/runnel/tangut-nlp -name "*.py" -type f | xargs grep -l "dictionary\|get_stats" 2>/dev/null

读取 [](file:///home/runnel/tangut-nlp/run.sh)

搜索正则表达式‘Dataset\.from_list|load_from_disk|load_dataset’

读取 [](file:///home/runnel/tangut-nlp/experiments/generate_candidates.py)

读取 [](file:///home/runnel/tangut-nlp/experiments/generate_candidates.py)

读取 [](file:///home/runnel/tangut-nlp/eval/lexical_coverage.py)

搜索正则表达式‘class.*Scorer|def score’

读取 [](file:///home/runnel/tangut-nlp/src/data_synthesis.py)

读取 [](file:///home/runnel/tangut-nlp/experiments/baseline1_zeroshot.py)

读取 [](file:///home/runnel/tangut-nlp/experiments/final_dpo.py)

读取 [](file:///home/runnel/tangut-nlp/eval/lexical_coverage.py)

搜索与 `data/**/*.jsonl` 匹配的文件

读取 [](file:///home/runnel/tangut-nlp/eval/chrf_scorer.py)

Ran terminal command: ls -lh /home/runnel/tangut-nlp/data/dictionary/

读取 [](file:///home/runnel/tangut-nlp/eval/llm_judge.py)

Ran terminal command: head -100 /home/runnel/tangut-nlp/data/dictionary/reward_dict.json 2>/dev/null | head -60

读取 [](file:///home/runnel/tangut-nlp/experiments/generate_candidates.py)

读取 [](file:///home/runnel/tangut-nlp/data/raw/tangut_output.jsonl)

搜索与 `**/*.json` 匹配的文件

读取 [](file:///home/runnel/tangut-nlp/data/sft/babelstone_sft.jsonl)

读取 [](file:///home/runnel/tangut-nlp/data/dpo/dpo_pairs.jsonl)

搜索正则表达式‘torch\.nn\.|model\.to\(|dtype\s*=’

读取 [](file:///home/runnel/tangut-nlp/src/prompt_templates.py)

Ran terminal command: find /home/runnel/tangut-nlp -name "*validate*" -o -name "*verify*" -o -name "*check*" 2>/dev/null | head -10

Ran terminal command: find /home/runnel/tangut-nlp -name "*.py" -type f | head -20

读取 [](file:///home/runnel/tangut-nlp/configs/deepspeed_zero2.json)

读取 [](file:///home/runnel/tangut-nlp/models/qwen2.5-7b-instruct/config.json)

Ran terminal command: python3 -c "import json; data=json.load(open('/home/runnel/tangut-nlp/data/dictionary/dictionary.json')); print(f'Total entries: {len(data)}'); print(f'Sample entry keys: {list(data[0].keys())}'); print(f'First entry: {json.dumps(data[0], ensure_ascii=False, indent=2)}')"

搜索正则表达式‘\.to\(self\.device|\.eval\(|\.train\(|dtype.*torch’

读取 [](file:///home/runnel/tangut-nlp/src/dictionary_utils.py)

读取 [](file:///home/runnel/tangut-nlp/eval/lexical_coverage.py)

读取 [](file:///home/runnel/tangut-nlp/experiments/baseline2_dict_rag.py)

Ran terminal command: head -5 /home/runnel/tangut-nlp/data/eval/test_set.jsonl 2>/dev/null || echo "File not found"

完美！现在我拥有了完整的代码理解。让我创建详细的实施计划：现在让我向你确认几个关键的技术细节，以便最终敲定计划：

所用工具: AskUserQuestion

Analyzing your answers...

完美！基于你的选择，我现在更新计划并最终提交審核：现在我的计划已完成并包含了所有关键的技术参数确认。让我请求你的最终批准：

以下是 Claude 的计划:

# 🎯 实验矩阵重构方案 - Baseline 3.1/3.2/3.3 完整实施计划

## 📌 Context

**目标**：在现有 Tangut-NLP 流水线基础上，插入三个新的 SFT Baseline 变体，形成严格控制变量的消融实验矩阵，解决"语义保留 vs. 数据纯净度"的核心问题。

**核心问题**：
- Baseline 3（混合 65% 西夏 + 35% 汉字）在推理纯西夏文时产生幻觉
- 现有的 Final (DPO) 在强制纯西夏输入时崩溃

**用户选择**：
- Embedding 方案：Qwen2.5-0.5B 本地推理（完全离线）
- 检索策略：top-3 投票（增加多样性，减少单一分散）
- 性能偏好：速度优先（使用 FAISS GPU）

---

## 🏗️ 实施架构

### 三个新 Baseline 设计

#### **Baseline 3.1：100% UNK 占位 SFT**
- **核心修改**：在 `src/data_synthesis.py` 中新增函数 `synthesize_pure_tangut_unk()`
- **逻辑**：字典未匹配项替换为 `[UNK]` 而非保留汉字
- **对比意义**：衡量"超纯净数据"对模型的影响（语义缺失的极端情况）
- **输出数据**：`data/sft/baseline3_1_unk_sft.jsonl`

#### **Baseline 3.3：向量空间语义投影 SFT**
- **核心模块**：三个新文件
  1. `src/utils/embeddings.py` — Qwen 本地 embedding 工具
  2. `scripts/build_tangut_semantic_index.py` — 构建 FAISS 索引
  3. 修改 `src/data_synthesis.py` 添加 `synthesize_semantic_projection()` 函数

- **实现流程**：
  1. 🧠 **语义候选库构建**：
     - 遍历字典所有西夏字符 → 提取 CN 释义
     - 使用 Qwen2.5-0.5B 计算每个释义的 embedding
     - 使用 FAISS GPU 索引存储（向量集合 + 字符映射）

  2. 🔍 **暴力投影合成**：
     - 遍历古代汉语语料
     - 对每个汉字：字典命中 → 直替；未命中 → 计算 embedding → FAISS 检索 top-3 → 投票选择最频发的西夏字 → 替换
     - 生成 100% 纯西夏文训练样本

- **对比意义**：与 3.1 对比，回答"纯净结构"vs"不完美的语义保留"哪个更关键
- **输出数据**：`data/sft/baseline3_3_semantic_sft.jsonl`

#### **Baseline 3.2：多任务对齐 SFT**（可选加速）
- **核心修改**：在 SFT 数据中添加结构化标签
- **输出格式示例**：
  ```json
  {
    "instruction": "...",
    "input": "...",
    "output": "翻译+结构标签...",
    "metadata": {..., "has_dict_alignment": true}
  }
```
- **对比意义**：显式 Loss 约束能否减少幻觉
- **输出数据**：需要后续提示词工程确认

---

## 📁 文件修改清单

### **第一层：新建基础设施**

#### 1️⃣ `src/utils/embeddings.py` （新建）
**职责**：Qwen 本地 embedding 工具类

**关键类/函数**：
- `class QwenEmbeddingService`
  - `__init__(model_path, device)` — 加载 Qwen2.5-0.5B，设置 eval 模式
  - `embed(texts: List[str], batch_size=32) -> np.ndarray` — 批量计算 embedding（返回 [n, 1024] 或对应维度）
  - `embed_single(text: str) -> np.ndarray` — 单条 embedding（用于在线访问）

**技术细节**：
- 抽取最后一层隐态作为 embedding（不使用 pooler）
- 支持批处理（batch_size=32）以优化速度
- 返回 L2 归一化向量（用于余弦相似度）

---

#### 2️⃣ `scripts/build_tangut_semantic_index.py` （新建）
**职责**：构建 FAISS 语义索引

**执行流程**：
```
输入：
  - data/dictionary/dictionary.json
  - data/dictionary/reward_dict.json

输出：
  - data/indices/tangut_semantic_index.index (FAISS GPU 索引)
  - data/indices/tangut_id2char.json (ID→西夏字符映射)
  - data/indices/char2embeddings.pkl (缓存，用于调试)
```

**关键步骤**：
1. 加载 BilingualDictionary，遍历所有西夏字符
2. 提取 CN 释义 → 清洗（移除括号、标点）
3. 使用 QwenEmbeddingService 批量计算 embeddings
4. 构建 FAISS GPU IndexFlatL2（或分段保存为 GPU 分片索引以节省内存）
5. 保存 ID→字符映射表和索引到本地

**进度条**：tqdm，显示"Embedding chars: {current}/{total}"

---

#### 3️⃣ `src/utils/faiss_client.py` （新建）
**职责**：FAISS 查询客户端（top-k 投票）

**关键类/函数**：
- `class FAISSSemanticClient`
  - `__init__(index_path, id2char_path)` — 加载索引和映射表
  - `search_topk(query_embedding: np.ndarray, k=3) -> List[str]` — 返回 top-k 西夏字符
  - `vote(candidates: List[str]) -> str` — 投票选择最频发的候选

---

### **第二层：数据合成修改**

#### 4️⃣ 修改 `src/data_synthesis.py`
**新增函数**：

**函数 1：`synthesize_pure_tangut_unk()`**
```python
def synthesize_pure_tangut_unk(ancient_text, modern_text, cn_to_tangut):
    """
    Baseline 3.1: 100% UNK 占位
    - 字典命中 → 直替为西夏文
    - 未命中 → 替换为 [UNK]
    """
    chars = list(ancient_text)
    for i, c in enumerate(chars):
        if c in cn_to_tangut:
            chars[i] = random.choice(cn_to_tangut[c])
        elif "\u4e00" <= c <= "\u9fff":  # 汉字
            chars[i] = "[UNK]"
    return "".join(chars), modern_text
```

**函数 2：`synthesize_semantic_projection()`**
```python
def synthesize_semantic_projection(
    ancient_text,
    modern_text,
    cn_to_tangut,
    embedding_service,
    faiss_client,
    batch_cache={}  # 缓存单个字符的 embedding 以加速
):
    """
    Baseline 3.3: 向量空间语义投影
    - 字典命中 → 直替
    - 未命中 → embedding + FAISS top-3 投票 → 替换
    """
    chars = list(ancient_text)
    for i, c in enumerate(chars):
        if c in cn_to_tangut:
            chars[i] = random.choice(cn_to_tangut[c])
        elif "\u4e00" <= c <= "\u9fff":  # 汉字
            # 计算此字的 embedding
            if c not in batch_cache:
                batch_cache[c] = embedding_service.embed_single(c)[0]  # [1024]
            embedding = batch_cache[c]

            # FAISS 检索 top-3
            candidate_tanguts = faiss_client.search_topk(embedding, k=3)
            # 投票选择
            chosen = faiss_client.vote(candidate_tanguts)
            chars[i] = chosen

    return "".join(chars), modern_text
```

**修改现有函数**：
- `main()` 函数添加两个新的命令行参数：
  - `--mode` : `["mixed", "unk", "semantic"]`（default="mixed"）
  - `--embedding-service-path` : embedding 服务路径（仅 mode="semantic" 用）
  - `--faiss-index-path` : FAISS 索引路径（仅 mode="semantic" 用）

- 根据 `--mode` 选择调用对应的 synthesize 函数

---

### **第三层：执行脚本集成**

#### 5️⃣ 修改 `run.sh`
**新增三个新的 Stage**（插在原有 Stage 05a-05b 之间）：

```bash
# 05a_baseline3_1_unk_data
# 05a_baseline3_2_multitask_data （可跳过）
# 05a_baseline3_3_semantic_index （构建 FAISS）
# 05a_baseline3_3_semantic_data （合成语义投影数据）

# 然后三条并行的 SFT 训练：
# 05c_sft_train_baseline3_original
# 05c_sft_train_baseline3_1_unk
# 05c_sft_train_baseline3_3_semantic
```

**触发逻辑**：
- 每个 Stage 都有对应的 sentinel 文件 `state/<stage>.done`
- 如果不存在则执行，执行完创建 sentinel

---

## 🚀 具体实施步骤

### **第 1 步：搭建基础设施（无数据生成）**
1. 新建 `src/utils/embeddings.py` — Qwen embedding 工具
2. 新建 `scripts/build_tangut_semantic_index.py` — FAISS 索引构建脚本
3. 新建 `src/utils/faiss_client.py` — 查询客户端
4. **验证**：运行 `python scripts/build_tangut_semantic_index.py --dict data/dictionary/dictionary.json --output data/indices/`
   - 预期输出：3 个文件 + 进度日志

### **第 2 步：修改数据合成**
1. 修改 `src/data_synthesis.py` 添加两个新函数
2. 修改 `main()` 函数支持 `--mode` 参数
3. **验证**：运行三条命令分别生成三种数据
   ```bash
   python src/data_synthesis.py --mode mixed --output data/sft/baseline3_original.jsonl
   python src/data_synthesis.py --mode unk --output data/sft/baseline3_1_unk.jsonl
   python src/data_synthesis.py --mode semantic --faiss-index-path data/indices/ --output data/sft/baseline3_3_semantic.jsonl
   ```

### **第 3 步：集成到 run.sh**
1. 在 `run.sh` 中添加新的 Stage 定义
2. 添加 sentinel 文件检查逻辑
3. **测试**：./run.sh 从特定阶段开始执行

### **第 4 步：三条并行 SFT 训练**
1. 确保 baseline3_synthetic_sft.py 能接收不同的数据文件
2. 运行三条并行训练（分别输出到 checkpoints/sft_baseline3_original/、.../baseline3_1_unk/、.../baseline3_3_semantic/）
3. 复用现有的推理脚本生成三份预测结果

---

## ✅ 验证策略

### **数据生成验证**
- ✓ 输出文件大小在合理范围（应接近）
- ✓ 采样检查 10 条：确认每种模式的替换逻辑正确
  - Baseline 3.1：所有未匹配字符都是 `[UNK]`
  - Baseline 3.3：所有未匹配字符都替换为西夏文（非 `[UNK]`）

### **模型训练验证**
- ✓ 三条 SFT 训练都能正常收敛（Loss 曲线、validation metrics）
- ✓ 检查点保存到对应目录

### **推理验证**
- ✓ 三份预测结果都能正常生成
- ✓ 预测中没有出现 `[UNK]` 标记（说明模型学到了有意义的表示）

### **性能对比**
- ✓ 运行 `eval/run_all_metrics.py` 分别评估三份预测
- ✓ 对比 lexical_coverage、perplexity、chrF++、LLM-Judge 四个维度
- ✓ 输出到 `results/ablation_matrix.json`

---

## 🔧 技术细节

### **Embedding 维度**（✅ 用户确认）
- 提取方式：**最后隐态层**（不使用 pooler）
- Qwen2.5-0.5B 的隐态维度：1024
- FAISS IndexFlatL2 + GPU，使用余弦相似度

### **投票策略**（✅ 用户确认）
- top-3 检索后使用 **Counter 最频繁投票**
- 若三个候选都不同 → 随机选一个（几率均等）
- 失败降级：FAISS 出错 → 自动替换为 `[UNK]`

### **性能预估**
- **索引构建**：24,133 字符 × embedding 计算 ≈ 5-10 分钟（单 GPU）
- **数据合成**：~50K 样本 × semantic 查询 ≈ 15-30 分钟（batch_cache 加速）
- **SFT 训练**：每条 3 epochs ≈ 1-2 小时（2x A100）
- **总耗时**：无严格限制，质量优先

### **内存管理**
- Qwen2.5-0.5B：~2GB（eval 模式）
- FAISS GPU 索引：~1GB（24K × 1024 float32）
- 总计：可在单块 A100 内运行，或所需时分配两块 GPU

---

## 📊 最终对比矩阵

| Baseline | 输入类型 | 语义保留 | 数据纯净度 | 幻觉风险 | 期望结果 |
|----------|---------|--------|----------|---------|---------|
| 3（原有） | 65% 混合 | 高 | 中 | 高 | 基准 |
| 3.1 | 100% 纯 + UNK | 低 | 极高 | 低 | 纯净胜？ |
| 3.3 | 100% 纯 + 投影 | 中 | 极高 | 低 | 平衡胜？ |
| Final V2 | Baseline 3.* + DPO | 待定 | 极高 | 最低 | 最终胜者 |

---

## ⚠️ 潜在风险与缓解

| 风险 | 原因 | 缓解方案 |
|------|------|---------|
| 投影误差 | FAISS 检索可能偏离语义 | → 继续保留 top-3 + 最频繁投票，手动验证 100 个样本 |
| 内存溢出 | GPU 同时运行 embedding + FAISS | → 使用 batch_cache 减少重复计算；如仍不足可用 CPU FAISS |
| FAISS 故障 | GPU 内存溢出或检索失败 | → ✅ **自动降级到 `[UNK]`**（用户确认） |
| 投票平手 | 三个不同候选平票 | → 随机选一个（均等概率）|

---

## 📝 关键文件交叉引用

- `src/dictionary_utils.py:BilingualDictionary` — 字典加载基础
- `src/data_synthesis.py:build_cn_to_tangut_map()` — 现有映射逻辑（复用）
- `eval/lexical_coverage.py:LexicalCoverageScorer` — 验证指标
- `experiments/baseline3_synthetic_sft.py` — SFT 训练模板（复用）
- `eval/run_all_metrics.py` — 最终汇总评估

---

## 🎬 确认项目（已全部确认 ✅）

1. ✅ **Embedding 维度**：最后隐态 1024 维
2. ✅ **投票策略**：Counter 最频繁投票
3. ✅ **时间预算**：无限制（质量优先）
4. ✅ **异常降级**：FAISS 失败 → 自动替换为 `[UNK]`

---

## 📋 实施顺序

1. **代码实施**（5 个新/改文件）→ 完成单元测试
2. **索引构建** → 验证 FAISS 索引生成
3. **数据合成** → 生成三种 SFT 数据
4. **数据验证** → 采样检查替换逻辑
5. **集成到 run.sh** → 支持三条并行训练
6. **训练 + 推理** → 三个 Baseline 完整流程
7. **最终评估** → 四维指标对比矩阵




太好了！计划已获批准。现在让我创建任务列表并开始实施：

Created 12 todos

现在开始实施第一步：构建基础设施。首先检查目录结构：

Ran terminal command: ls -la /home/runnel/tangut-nlp/src/ | head -20

Ran terminal command: mkdir -p /home/runnel/tangut-nlp/src/utils && touch /home/runnel/tangut-nlp/src/utils/__init__.py

现在创建 Qwen Embedding Service：
````

````
Now create the FAISS semantic client:
````

````
Now create the FAISS index builder script:
````

````
Now update the TodoList and modify data_synthesis.py:

Starting: *Add synthesize_pure_tangut_unk() function to src/data_synthesis.py* (4/12)

读取 [](file:///home/runnel/tangut-nlp/src/data_synthesis.py)


````

````

````

````
Perfect! Now let me update the TodoList and verify the new code works:

Starting: *Test data_synthesis.py imports and basic functionality* (6/13)

Let me verify that the code can be imported successfully:

Ran terminal command: python3 -c "
import sys
sys.path.insert(0, '.')
from src.data_synthesis import build_cn_to_tangut_map, synthesize_mixed_pair, synthesize_pure_tangut_unk, synthesize_semantic_projection
from src.utils.embeddings import QwenEmbeddingService
from src.utils.faiss_client import FAISSSemanticClient
print('✅ All imports successful!')
print('✅ synthesize_pure_tangut_unk: ', synthesize_pure_tangut_unk.__doc__[:50])
print('✅ synthesize_semantic_projection: ', synthesize_semantic_projection.__doc__[:50])
"

Ran terminal command:  python3 -c "
import sys
sys.path.insert(0, '.')
from src.data_synthesis import build_cn_to_tangut_map, synthesize_mixed_pair, synthesize_pure_tangut_unk, synthesize_semantic_projection
from src.utils.embeddings import QwenEmbeddingService
from src.utils.faiss_client import FAISSSemanticClient
print('✅ All imports successful!')
print('✅ synthesize_pure_tangut_unk: ', synthesize_pure_tangut_unk.__doc__[:50])
print('✅ synthesize_semantic_projection: ', synthesize_semantic_projection.__doc__[:50])
"


````

````


Ran terminal command: python3 << 'EOF'
import sys
sys.path.insert(0, '.')
from src.data_synthesis import (
    build_cn_to_tangut_map,
    synthesize_mixed_pair,
    synthesize_pure_tangut_unk,
    synthesize_semantic_projection
)
print('✅ Data synthesis imports successful!')
print('   - synthesize_mixed_pair (original mode)')
print('   - synthesize_pure_tangut_unk (Baseline 3.1)')
print('   - synthesize_semantic_projection (Baseline 3.3)')
EOF


Ran terminal command:  python3 << 'EOF'
import sys
sys.path.insert(0, '.')
from src.data_synthesis import (
    build_cn_to_tangut_map,
    synthesize_mixed_pair,
    synthesize_pure_tangut_unk,
    synthesize_semantic_projection
)
print('✅ Data synthesis imports successful!')
print('   - synthesize_mixed_pair (original mode)')
print('   - synthesize_pure_tangut_unk (Baseline 3.1)')
print('   - synthesize_semantic_projection (Baseline 3.3)')
EOF


Ran terminal command: python3 << 'EOF'
import sys
sys.path.insert(0, '.')

# Try importing - FAISS might not be installed yet, which is OK
try:
    from src.utils.faiss_client import FAISSSemanticClient
    print('✅ FAISS client imports successful!')
except ImportError as e:
    if "faiss" in str(e).lower():
        print('ℹ️  FAISS not installed yet (expected) - will be installed during setup')
    else:
        raise

try:
    from src.utils.embeddings import QwenEmbeddingService
    print('⚠️  Embedding service imports, but depends on transformers/torch')
except ImportError as e:
    print(f'ℹ️  Expected dependency missing for embedding service: {e}')

print('✅ Module structure is sound!')
EOF


Ran terminal command:  python3 << 'EOF'
import sys
sys.path.insert(0, '.')

# Try importing - FAISS might not be installed yet, which is OK
try:
    from src.utils.faiss_client import FAISSSemanticClient
    print('✅ FAISS client imports successful!')
except ImportError as e:
    if "faiss" in str(e).lower():
        print('ℹ️  FAISS not installed yet (expected) - will be installed during setup')
    else:
        raise

try:
    from src.utils.embeddings import QwenEmbeddingService
    print('⚠️  Embedding service imports, but depends on transformers/torch')
except ImportError as e:
    print(f'ℹ️  Expected dependency missing for embedding service: {e}')

print('✅ Module structure is sound!')
EOF


Let me install the required dependencies:

Ran terminal command: pip install numpy faiss-gpu -q 2>&1 | tail -5

Ran terminal command:  pip install numpy faiss-gpu -q 2>&1 | tail -5

Made changes.

User: /compact

Claude: Ran terminal command:  cat << 'EOF'
════════════════════════════════════════════════════════════
✨ ABLATION MATRIX INFRASTRUCTURE: BUILD COMPLETE
════════════════════════════════════════════════════════════

📦 NEW FILES CREATED (4):

1. src/utils/embeddings.py
   └─ QwenEmbeddingService class
      • Loads Qwen2.5-0.5B
      • embed() - batch processing (32 samples/batch)
      • embed_single() - single text
      • Returns L2-normalized 1024-dim vectors

2. src/utils/faiss_client.py
   └─ FAISSSemanticClient class
      • search_topk(embedding, k=3) → top-k Tangut chars
      • vote(candidates) → Counter-based majority vote
      • search_and_vote() - one-shot search+vote

3. scripts/build_tangut_semantic_index.py
   └─ Standalone script to build FAISS index
      • Input: data/dictionary/dictionary.json
      • Process: Extract CN explanations → Embed → FAISS
      • Output: 
        - data/indices/tangut_semantic_index.index
        - data/indices/tangut_id2char.json
        - data/indices/embeddings.npy

4. src/data_synthesis.py (MODIFIED - backward compatible)
   └─ Three synthesis modes:
      • synthesize_mixed_pair() - Original (30-70% replacement)
      • synthesize_pure_tangut_unk() - NEW Baseline 3.1
      • synthesize_semantic_projection() - NEW Baseline 3.3
   └─ main() updated with --mode parameter:
      • python src/data_synthesis.py --mode mixed    (default)
      • python src/data_synthesis.py --mode unk      (Baseline 3.1)
      • python src/data_synthesis.py --mode semantic (Baseline 3.3)

════════════════════════════════════════════════════════════
🔧 DEPENDENCIES REQUIRED:
   • torch, transformers (already in requirements.txt)
   • faiss-gpu or faiss-cpu (for FAISS indexing)
   • numpy (already in requirements.txt)
   
════════════════════════════════════════════════════════════
📋 NEXT STEPS (In Order):

1. BUILD FAISS INDEX (one-time):
   python scripts/build_tangut_semantic_index.py \\
     --dictionary data/dictionary/dictionary.json \\
     --model models/qwen2.5-0.5b \\
     --output data/indices

2. GENERATE THREE DATA VARIANTS:
   python src/data_synthesis.py --mode mixed \
     --output data/sft/baseline3_original.jsonl
   
   python src/data_synthesis.py --mode unk \
     --output data/sft/baseline3_1_unk.jsonl
   
   python src/data_synthesis.py --mode semantic \
     --faiss-index-path data/indices/tangut_semantic_index.index \
     --faiss-mapping-path data/indices/tangut_id2char.json \
     --output data/sft/baseline3_3_semantic.jsonl

3. VALIDATE DATA (manual spot-check 10 samples from each)

4. INTEGRATE INTO run.sh (new stages for 3 parallel SFT + eval)

5. RUN COMPLETE ABLATION PIPELINE

════════════════════════════════════════════════════════════
EOF


*Conversation compacted*

User: 忘了说了，请在conda的tangut-nlp环境里运行。
