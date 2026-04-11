# Tangut-NLP：极低资源西夏文短标题翻译

本仓库研究的是一个非常具体、也非常难的任务：在只有 `491` 条真实西夏文-中文平行对的条件下，如何把西夏文短文本，尤其是经名、书名、目录项这类**短标题**，翻译成尽可能准确、紧凑的现代中文标题。

这不是一个“通用西夏文机器翻译”仓库。按当前证据，它更准确的定位是：

- **任务边界**：西夏文短文本 / 题名体翻译
- **核心资源**：`491` 条真实平行对，划分为 `400/41/50` 的 train/dev/test
- **主要技术路线**：强提示词推理、合成数据 SFT、DPO 偏好优化、以及最终的多候选裁决工作流
- **当前最强结果**：不是单一路线通杀，而是“frontier 强提示词 + 本地结构化训练候选”的互补

如果只看一句话结论：

- **最强单模型**是 `frontier_deepseek_v32_fewshot_cot`：reference-aware overall `2.80/5`，`12/50` exact match，`24/27` 标题后缀保持，`0/50` 污染。
- **最强本地可训练单模型**是 `final_gap04_multitask_sigmoid`：chrF++ `30.01`，reference-aware overall `2.22/5`，`5/50` exact match，长度比 `1.12`。
- **最强整体工作流**是 `hybrid_multi3_catalog_gpt54`：chrF++ `34.87`，reference-aware overall `3.02/5`，`14/50` exact match，`26/27` 标题后缀保持，`0/50` 污染。

## 1. 当前项目状态

### 1.1 现在最可信的故事线

这套实验现在支持的结论，不再是“只要本地训练就能压过所有 prompt-only 系统”，而是下面这条更准确的叙事：

1. **强 SOTA 中文大模型 + 强提示词，确实已经很强。**
2. **但结构化本地训练仍然有价值，因为它会产生 frontier 模型没有的保守候选。**
3. **最好的结果来自互补，而不是来自押注单一路线。**
4. **DPO 不是天然无效，但旧版 DPO 的确训崩过；修复关键不在“多训几轮”，而在“换对底座 + 过滤脏偏好对”。**

### 1.2 关键结果总表

| 系统 | 类型 | chrF++ | Ref-aware Overall | Exact Match | 污染 | 说明 |
|---|---|---:|---:|---:|---:|---|
| `baseline3_2_multitask` | 最强纯 SFT 底座 | 25.99 | 1.98 | 5/50 | 1/50 | 结构化 SFT 仍是最好的纯监督底座 |
| `frontier_deepseek_v32_fewshot_cot` | 最强单模型 | 28.54 | 2.80 | 12/50 | 0/50 | 强 proprietary prompt-only 上界 |
| `final_gap04_multitask_sigmoid` | 最强本地 DPO | 30.01 | 2.22 | 5/50 | 2/50 | 最强本地可训练单模型 |
| `open_hybrid_heuristic_guarded` | 开放式 3-way selector | 32.62 | 2.90 | 12/50 | 0/50 | 确定性 heuristic selector，不做自由改写 |
| `hybrid_multi3_catalog_gpt54` | 最强整体工作流 | 34.87 | 3.02 | 14/50 | 0/50 | 当前仓库最强非参考系统 |

这里的“Ref-aware Overall”来自 reference-aware judge：它同时看西夏文原文、参考译文和候选译文，比旧版 reference-free LLM judge 更可信。

## 2. 方法谱系

当前仓库的主要路线可以概括为四层：

| 路线 | 代表方法 | 现在的作用 |
|---|---|---|
| Prompt-only（本地） | `baseline1`、`baseline2`、`baseline2_1_cot` | 证明单纯字典注入和 CoT 只能走到哪里 |
| Prompt-only（frontier） | `frontier_deepseek_v32_fewshot_cot` | 回答“强 SOTA + 强 prompt 会不会已经够了” |
| Synthetic SFT | `baseline3`、`baseline3_1_unk`、`baseline3_2_multitask`、`baseline3_3_semantic` | 找到最强本地可训练底座 |
| Preference + Workflow | `final_v2`、`final_gap04_multitask_sigmoid`、`hybrid_*` | 区分“坏 DPO”和“修好的 DPO”，并验证互补工作流 |

## 3. 导师最关心的两个问题

### 3.1 之前的 DPO 为什么训崩了？现在怎么修复的？

短答案：**旧版 DPO 失败，不是因为 DPO 这个方法在西夏文任务上天然无效，而是因为旧管线同时犯了几处关键的结构性错误。**

#### 旧版 DPO 为什么会崩

1. **底座选错了。**
   旧版 `final_v2` 的底座选择逻辑只在 `mixed / unk / semantic` 三个 SFT 底座里按 chrF 自动选最优，**根本没把最强的 `multitask` 底座放进选择池**。结果 legacy DPO 实际上是从 `UNK` 底座起训，而不是从最强底座起训。

2. **偏好对质量不够干净。**
   当前审计显示，原始 `data/dpo/dpo_pairs.jsonl` 一共有 `3867` 对，其中：
   - `3` 对是 chosen/rejected 完全相同的重复对
   - `3` 对 reward 是 non-finite
   - 全体样本里，reward-chosen 比 reward-rejected 更接近 gold proxy 的比例只有 `71.3%`
   - 对于最危险的低 gap 区间 `(0.05, 0.10]`，这个比例只有 `62.3%`

3. **gap 门槛太松，噪声标签直接进训练。**
   旧 DPO 的最小 gap 只有 `0.05`。这意味着大量“chosen 其实也没比 rejected 好多少”的样本被当成了硬偏好监督，训练会把模型往错误方向推。

4. **优化目标和真正任务目标不完全对齐。**
   legacy reward 是：

   ```text
   reward = lexical_coverage - 0.01 * log(perplexity)
   ```

   这个目标对“词典命中”和“现代中文 LM 友好度”有偏好，但我们真正要的任务是**保守的历史短标题重建**。这两者不是一回事。

#### 旧版 DPO 崩成什么样

| 模型 | chrF++ | Ref-aware Overall | Exact Match | 污染 | Length Ratio |
|---|---:|---:|---:|---:|---:|
| `baseline3_1_unk` | 22.11 | 1.82 | 3/50 | 3/50 | 1.05 |
| `baseline3_2_multitask` | 25.99 | 1.98 | 5/50 | 1/50 | 1.10 |
| `final_v2` | 19.39 | 1.68 | 2/50 | 9/50 | 1.88 |
| `final` | 9.23 | - | 0/50 | 17/50 | 4.74 |

所以旧版不是“略差一点”，而是典型的**过扩写、污染、标题形态失控**。

#### 现在是怎么修复的

修复也不是一句“把参数调好了”，而是明确改了三件事：

1. **把 DPO 移到真正最强的 `multitask` SFT 底座上。**
2. **先过滤偏好对，再训练。**
   用 [`scripts/filter_dpo_pairs.py`](scripts/filter_dpo_pairs.py) 做：
   - non-finite 剔除
   - chosen=rejected 剔除
   - dedupe
   - reward gap 过滤
3. **把 gap 拉高到真正能区分“好坏候选”的区间。**
   - `gap >= 0.2` 保留 `1996` 对
   - `gap >= 0.4` 只保留 `498` 对
   - 但这 `498` 对的 proxy 质量明显更高，chosen better rate 提升到 `81.5%`

#### 修复后为什么稳定了

最关键的发现是：**真正起作用的是“高质量底座 + aggressive gap filtering”，而不是 PPL 系数怎么微调。**

`results/analysis/reward_weight_ablation.json` 显示，当 reward 里的 PPL 权重在 `0.0 / 0.005 / 0.01 / 0.02` 之间变化时：

- 相对当前设置的 pair order **sign flip = 0**
- `gap >= 0.4` 的保留对数只在 `498` 到 `504` 之间轻微波动

换句话说，当前修复的主因不是“我们终于把 reward 写对了”，而是**我们终于把脏偏好对挡在训练外面了**。

#### 修复后的结果

| 模型 | chrF++ | Ref-aware Overall | Exact Match | 污染 | Length Ratio |
|---|---:|---:|---:|---:|---:|
| `baseline3_2_multitask` | 25.99 | 1.98 | 5/50 | 1/50 | 1.10 |
| `final_gap02_multitask_sigmoid` | 20.93 | 2.22 | 5/50 | 0/50 | 1.74 |
| `final_gap04_multitask_sigmoid` | **30.01** | **2.22** | **5/50** | 2/50 | **1.12** |
| `final_gap04_multitask_robustwpo` | 27.52 | **2.22** | **5/50** | 2/50 | 1.26 |

现在最稳的结论是：

- **DPO 不是完全没用**
- **但 legacy DPO 的确是坏设计**
- **DPO 真正被修好的关键，是换到 `multitask` 底座并把偏好对 gap 严格过滤到 `0.4` 左右**

### 3.2 为什么不直接用 SOTA 中文大模型加上足够强的提示词？

短答案：**我们已经这么做了，而且它确实很强；但它还不是整个问题的终点。**

#### 我们不是拿一个弱 prompt 去敷衍这个问题

本仓库专门做了一个 reviewer-style 的强基线：[`experiments/frontier_openrouter_dict.py`](experiments/frontier_openrouter_dict.py)

配置是：

- 模型：`deepseek/deepseek-v3.2`
- 提示模式：`fewshot`
- 词典输入：逐段 gloss，不只是原文
- 解码：`temperature=0.0`
- few-shot 示例：固定使用 dev set 中 4 个例子
  - `佛説大人八覺經`
  - `佛説菩薩行修經`
  - `金光明總持經`
  - `摩訶般若波羅蜜多心經`

它的系统提示不是泛泛地说“请翻译”，而是明确加入了这些强约束：

```text
你是一位非常严格的西夏文书名/短标题翻译专家。
你的任务不是写解释、摘要或现代白话句子，而是把西夏文短文本
还原成尽可能准确、紧凑的现代中文标题。

必须同时满足：
1. 优先输出标题/书名体，而不是解释性句子
2. 只能根据给定西夏文和词典释义做判断；不能虚构上下文
3. 多义词必须全局消歧，宁可保守，也不要把多个释义拼成长句
4. 不要机械照抄泛化佛经名
5. 优先保持 經、論、記、疏、頌、儀、義、傳、錄、贊、序、品、文、觀、根、次 等结尾
6. 最终输出应尽量短，接近真实标题长度
7. 只输出标题本身
```

用户提示里还额外要求模型：

- 先识别专名、音译佛典词、结构词、题名后缀
- 再做全局消歧
- 如果多个候选都可能，优先选择**更像真实中文书名、且更短更克制**的一项

这已经不是“普通 prompt engineering”，而是**词典 grounding + 标题体约束 + anti-expansion 规则 + fixed few-shot exemplar** 的强提示词方案。

#### 强 prompt 的结果是什么

结果必须实事求是地承认：**它真的强。**

| 系统 | chrF++ | Ref-aware Overall | Exact Match | 标题后缀保持 | 污染 |
|---|---:|---:|---:|---:|---:|
| `baseline2` | 6.13 | 1.50 | 0/50 | 3/27 | 0/50 |
| `baseline2_1_cot` | 7.59 | - | 1/50 | 3/27 | 5/50 |
| `frontier_deepseek_v32_fewshot_cot` | 28.54 | **2.80** | **12/50** | **24/27** | **0/50** |

所以对导师这个问题，最诚实的回答不是“我们没试过”，而是：

1. **我们试过，而且它是当前最强单模型。**
2. **因此论文不能再讲成‘prompt-only 一定不行’。**

#### 那为什么还要做本地训练？

因为 frontier 强提示词虽然强，但它有自己的系统性失败模式：**过度正规化、过度依赖常见中文题名先验、在极短标题上容易 hallucinate 或过拟合成熟佛典名。**

几个仓库里已经明确记录的例子：

| 参考 | Frontier 输出 | 本地 / 工作流输出 | 现象 |
|---|---|---|---|
| `五部經` | `五類經` | `五部經` | frontier 过度正规化 |
| `番言金剛王乘根` | `番` | `番言金剛王乘根` | frontier 极短输入时截断 |
| `正理滴之句義顯具` | `德宜點義句顯宣論` | `正理滴之句義顯了` | frontier 被词典表层释义带偏 |

这正是本地结构化训练的价值：

- 它不一定在“总体流畅性”上压过 frontier
- 但它能提供一批**更保守、更贴目录体、更少现代化正规化**的候选
- 这些候选和 frontier 的错误模式不同
- 一旦做 3-way adjudication，整体效果就会超过 frontier 单模型

这也是为什么当前最强结果不是单模型，而是：

| 工作流 | chrF++ | Ref-aware Overall | Exact Match |
|---|---:|---:|---:|
| `frontier_deepseek_v32_fewshot_cot` | 28.54 | 2.80 | 12/50 |
| `open_hybrid_heuristic_guarded` | 32.62 | 2.90 | 12/50 |
| `hybrid_multi3_catalog_gpt54` | **34.87** | **3.02** | **14/50** |

所以对“为什么不直接用 SOTA 中文大模型 + 强 prompt”的最终回答应该是：

- **如果你只想要最强单模型上界，可以，frontier prompt-only 就是现在最强单模型。**
- **但如果你关心的是更稳的历史题名重建工作流，本地结构化训练仍然有不可替代的互补价值。**

## 4. 为什么现在主要看 chrF++ 和 reference-aware judge

这个任务里，很多看起来合理的 reference-free 指标其实会误导结论。

最典型的 sanity check 是 `human_reference`：把测试集参考译文直接喂回评测管线，理论上它应该是最优。

但当前结果是：

- `chrF++ = 100.00`
- `lexical coverage = 0.5733`
- `perplexity = 1646139.34`

这说明：

1. **词典覆盖率不等于真实翻译质量**
2. **现代中文 LM 的 PPL 会惩罚题名体、历史专名和音译链条**
3. **所以 lexical coverage / PPL / 旧版 reference-free judge 更适合做诊断，不适合作为 headline model selection signal**

当前最可信的主证据是：

- chrF++
- exact match
- contamination
- 标题后缀保持
- reference-aware judge

## 5. 仓库里最该看的文件

如果你想快速对齐当前项目状态，优先看下面这些文件：

- [`docs/NARRATIVE_REPORT.md`](docs/NARRATIVE_REPORT.md)：当前论文故事线的完整叙述
- [`docs/findings.md`](docs/findings.md)：关键发现的压缩日志
- [`docs/EXPERIMENT_LOG.md`](docs/EXPERIMENT_LOG.md)：实验块级别的结果记录
- [`experiments/frontier_openrouter_dict.py`](experiments/frontier_openrouter_dict.py)：强 frontier prompt-only 基线
- [`experiments/final_dpo.py`](experiments/final_dpo.py)：DPO 训练脚本
- [`scripts/filter_dpo_pairs.py`](scripts/filter_dpo_pairs.py)：DPO 对过滤逻辑
- [`experiments/hybrid_adjudication_multicandidate.py`](experiments/hybrid_adjudication_multicandidate.py)：3-way / 4-way catalog hybrid
- [`paper/README.md`](paper/README.md)：论文 LaTeX 编译说明

## 6. 最小复现入口

### 6.1 重新计算当前诊断

```bash
python3 scripts/analyze_snapshot.py
```

### 6.2 跑最强 frontier prompt-only 基线

```bash
python3 experiments/frontier_openrouter_dict.py \
  --prompt-mode fewshot \
  --model deepseek/deepseek-v3.2 \
  --output results/frontier_deepseek_v32_fewshot_cot/predictions.jsonl \
  --method-name frontier_deepseek_v32_fewshot_cot
```

### 6.3 过滤 DPO 对并重训修复版 DPO

```bash
python3 scripts/filter_dpo_pairs.py \
  --input data/dpo/dpo_pairs.jsonl \
  --output data/dpo/dpo_pairs_gap04.jsonl \
  --min-gap 0.4 \
  --dedupe
```

```bash
python3 experiments/final_dpo.py \
  --dpo-data data/dpo/dpo_pairs_gap04.jsonl \
  --sft-model checkpoints/sft_multitask/merged \
  --output-dir checkpoints/dpo_gap04_multitask_sigmoid \
  --loss-type sigmoid
```


## 7. 当前最稳的结论边界


- 我们**已经证明**：强 SOTA 中文大模型 + 强 prompt 在这个任务上非常强。
- 我们**已经证明**：legacy DPO 确实会崩，而且崩因可以被定位到错误底座和脏偏好对。
- 我们**已经证明**：换到 `multitask` 底座并严格做 gap 过滤后，DPO 可以从坏结果变成强本地模型。
- 我们**已经证明**：frontier 与本地结构化训练是互补关系，3-way workflow 明显优于 frontier 单模型。
- 我们**还没有证明**：本地开放模型已经全面压过最强 proprietary 单模型。
- 我们**也没有打算宣称**：这个仓库已经解决了通用西夏文机器翻译。

