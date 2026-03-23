# Tangut NLP: 极低资源西夏文机器翻译研究

本仓库对应一项面向极低资源语言的机器翻译研究。核心问题是：在几乎不存在高质量平行语料的前提下，是否可以仅依赖字典知识、单语语料与大语言模型，实现可评估、可复现、可迭代优化的西夏文到中文翻译系统。

本文档采用论文导向组织方式，强调研究问题、实验设计、对比关系与结论可解释性，而非工程部署细节。

## 1. 研究问题与动机

西夏文属于典型的超低资源场景，主要挑战包括：

1. 缺乏可直接监督训练的大规模平行语料。
2. 书写系统与现代中文差异显著，字面映射不足以保证句法可迁移。
3. 评估时难以依赖传统 BLEU 与人工大规模标注。

本研究围绕一个中心假设展开：

在输入端强制“纯西夏化”后，模型性能变化由两类因素主导：
1. 语义保留程度。
2. 结构纯净程度。

## 2. 方法框架与实验矩阵

我们构建了三条递进流派，形成统一消融矩阵。

### 2.1 推理层过渡

1. Baseline 1: Zero-shot
- 直接对西夏文输入进行指令翻译。
- 目标：估计大模型对西夏文 Unicode 的原生可迁移能力。

2. Baseline 2: Dictionary-Augmented RAG
- 以字典释义作为外部知识注入，不更新模型参数。
- 目标：测量“仅靠符号注入”的上限。

3. Baseline 2.1: RAG + 三步 CoT
- Prompt 显式拆分为“字面对齐-直译-意译重写”。
- 变量：仅改变推理提示词结构。

### 2.2 数据与 SFT 过渡

1. Baseline 3: 混合输入 SFT
- 合成阶段保留部分汉字，输入并非完全纯西夏。
- 作用：作为“混合输入对照组”。

2. Baseline 3.1: 100% UNK 纯化 SFT
- 未命中字典项统一替换为 [UNK]。
- 优点：输入纯净。
- 风险：语义严重断裂。

3. Baseline 3.3: 向量空间语义投影 SFT
- 未命中字典项不再置为 [UNK]，而是以语义向量近邻检索方式强制投影到可替换西夏字符。
- 与 3.1 形成对照：
  语义断裂替换 vs 语义扭曲替换。

### 2.3 架构与对齐过渡

1. Baseline 3.2: 多任务结构化 SFT
- 在目标端引入结构化字段约束（例如 dict_match 与 literal 段落）。
- 目的：抑制幻觉并增强对齐显式性。

2. Final（原始 DPO）
- 在非纯净底座上直接做偏好对齐，表现不稳定。
- 角色：失败路径记录。

3. Final V2（受控 DPO）
- 从 3.1、3.2、3.3 中选性能最佳底座，再构建偏好对执行 DPO。
- 目标：在保持输入纯净性的同时恢复语义与流畅度。

## 3. 评估协议

在无高质量黄金参考对齐条件下，采用四维评估矩阵。

1. Lexical Coverage（忠实度）
- 度量输出中命中字典语义集合的比例。
- 反映“是否覆盖了源句关键语义”。

2. Perplexity（流畅度约束）
- 使用轻量中文语言模型计算困惑度。
- 评价句法自然性与可读性。

3. chrF++（字符级软匹配）
- 对古文场景较 BLEU 更稳健。
- 反映输出与参考在字符形态层面的接近程度。

4. LLM-as-a-Judge（语义完整与语句通顺）
- 对每个候选给出双维评分：语义完整性、语言流畅性。

在 DPO 候选构建中，当前实现使用如下奖励形式进行排序：

$$
R(y)=\alpha\cdot\text{LexicalCoverage}(y)-\beta\cdot\log\left(\text{PPL}(y)+\epsilon\right)
$$

其中 $y$ 为候选翻译，当前配置为 $\alpha=1.0,\beta=0.01,\epsilon=10^{-8}$。每个输入采样 $N=5$ 个候选，按奖励排序后取最高分样本作为 Chosen、最低分样本作为 Rejected，并设置最小 reward gap 过滤阈值（0.05）。

## 4. 主要实现映射

为便于论文复现，给出研究模块与代码入口对应关系：

1. 实验入口：experiments/
2. 数据合成与字典处理：src/
3. 指标计算：eval/
4. 训练配置：configs/
5. 结果归档：results/
6. 检查点：checkpoints/
7. 阶段状态与断点：state/

## 5. 复现实验（论文附录风格）

建议硬件：2 x A100（80GB）或同等级显存设备。

最小复现流程：

1. 环境准备

```bash
chmod +x scripts/*.sh
./scripts/setup_env.sh
```

2. 全流程运行

```bash
./run.sh
```

3. 汇总对比结果

```bash
python eval/aggregate_results.py
```

4. 关键输出
- 各基线指标：results/[experiment]/metrics.json
- 总表：results/comparison.csv 与 results/comparison.json
- 可视化：results/comparison.png 与 results/comparison_radar.png

## 6. 现有结果分析

本节基于当前仓库中的已落盘结果进行陈述，数据来源为 [results/comparison.csv](results/comparison.csv) 与 [results/comparison.json](results/comparison.json)。

图表引用：

1. 总览柱状图：[results/comparison.png](results/comparison.png)
2. 全方法雷达图：[results/comparison_radar.png](results/comparison_radar.png)
3. 分方法雷达图：[results/comparison_radar_each.png](results/comparison_radar_each.png)
4. 相对人类基线雷达图：[results/comparison_radar_relative.png](results/comparison_radar_relative.png)

### 6.1 核心观测

1. 忠实度（Lexical Coverage）最高的是 Baseline 2.1（0.8723），其次是 Baseline 2（0.8187）。
2. 流畅度（PPL，越低越好）最优的是 Baseline 1（20.21），而非后续微调与对齐方法。
3. chrF++ 最高的是 Baseline 3.2（25.99），明显高于 Baseline 2 系列与 Final 系列。
4. LLM Judge 的语义与流畅双分最高均出现在 Baseline 2（2.30 / 2.40）。
5. Final 与 Final V2 并未成为综合最优：
  Final 的 PPL 极高（158427.71）且 Judge 分数最低组；
  Final V2 的 PPL 同样极高（314577.16），虽 chrF++ 有提升，但 Judge 仍偏低。

### 6.2 与研究预期的不一致点

按原研究假设，Final V2（受控 DPO）应在“忠实度-流畅度-语义完整性”之间取得更优折中；但当前结果呈现如下偏差：

1. 预期 DPO 改善语义与流畅：未观察到。
  Final 与 Final V2 的 LLM Judge 分数分别为 (1.40, 1.42) 与 (1.52, 1.76)，均未超过 Baseline 2。

2. 预期 3.3 优于 3.1：未稳定成立。
  在当前结果中，Baseline 3.1 在 chrF++（22.11）与 Judge（1.66/1.96）上均优于 Baseline 3.3（17.03，1.46/1.84）。

3. 预期“更复杂方法整体更优”：未成立。
  Baseline 2/2.1 在忠实度与 Judge 上更强；Baseline 3.2 在 chrF++ 上更强；不存在单一统治解。

### 6.3 当前强化学习（DPO）方案说明

为避免术语歧义，本项目“强化学习阶段”采用的是离线偏好优化（DPO），而非在线 RLHF。完整流程如下：

1. 底座选择：在 mixed / unk / semantic 三个 SFT 底座中，按 chrF++ 自动选择最佳底座。
2. 候选采样：对每个训练输入从 SFT 底座采样 $N=5$ 候选（temperature=0.8, top-p=0.95）。
3. 奖励打分：对每个候选计算 Lexical Coverage 与 PPL，并按上式得到 reward。
4. 偏好对构造：选择 best-vs-worst 形成 (chosen, rejected)，并丢弃 reward gap 小于 0.05 的样本。
5. DPO 训练：使用同一 SFT 模型作为 policy/ref 初始点，进行偏好优化，得到 Final V2。

该流程的目标是让模型在“词典忠实度”与“语言流畅性”之间做可控折中。

### 6.4 噪声来源猜测（基于当前快照）

结合当前结果与偏好数据统计，噪声可能来自以下层面：

1. 偏好标签噪声：在现有 dpo_pairs 中存在少量异常条目（例如 non-finite reward、chosen/rejected 文本相同），会直接引入无效学习信号。
2. 奖励尺度噪声：PPL 分布长尾明显，即便使用 log 压缩，固定系数 $\beta$ 仍可能在子域间失衡。
3. 采样方差：每条输入仅采样 5 个候选，易出现“相对最优但绝对质量一般”的伪偏好。
4. 目标错位噪声：DPO 训练目标基于 Lexical+PPL，而最终报告还依赖 chrF 与 LLM Judge，优化目标与评测目标并不完全一致。
5. 评测噪声：LLM Judge 受 API 稳定性与解析策略影响，可能引入额外方差。

据此，当前“Final V2 未优于最佳 SFT”的现象可被解释为：偏好信号质量不足以稳定驱动 DPO 获得跨指标一致增益。

### 6.5 各指标记录表（当前快照，n=50）

| Experiment | Lexical Coverage ↑ | PPL ↓ | chrF++ ↑ | LLM Semantic ↑ | LLM Fluency ↑ |
|---|---:|---:|---:|---:|---:|
| baseline1 | 0.1165 | 20.21 | 0.19 | 1.00 | 1.56 |
| baseline2 | 0.8187 | 4474.44 | 6.13 | 2.30 | 2.40 |
| baseline2_1_cot | 0.8723 | 18119.16 | 7.59 | 2.30 | 1.88 |
| baseline3 | 0.3192 | 298684.91 | 17.85 | 1.42 | 1.88 |
| baseline3_1_unk | 0.3514 | 11031.18 | 22.11 | 1.66 | 1.96 |
| baseline3_2_multitask | 0.4839 | 3322.93 | 25.99 | 1.70 | 2.26 |
| baseline3_3_semantic | 0.3723 | 17064.38 | 17.03 | 1.46 | 1.84 |
| final | 0.3913 | 158427.71 | 9.23 | 1.40 | 1.42 |
| final_v2 | 0.4224 | 314577.16 | 19.39 | 1.52 | 1.76 |
| human_reference | 0.5733 | 1646139.34 | 100.00 | 2.24 | 2.34 |

注：该表仅陈述当前仓库快照，不对未来重训后的变化作外推。

### 6.6 真实译文基线（human_reference）与反直觉现象

为了验证“真实翻译并不一定在所有自动指标上都高分”，我们新增了 human_reference 基线：将测试集参考译文直接作为预测输入同一评估管线。

当前结果如下：

1. chrF++ = 100.00（与参考完全一致，符合预期）。
2. Lexical Coverage = 0.5733（并非满分）。
3. PPL = 1646139.34（显著高于所有 baseline）。
4. LLM Judge = 2.24 / 2.34（高于大多数 SFT/DPO 方法，但未显著高于 Baseline 2 的流畅度分）。

这说明：

1. 词典覆盖指标并不等价于“真实翻译质量”，会受到词典覆盖范围与释义粒度限制。
2. 以现代通用 LM 计算的 PPL 不一定适配题名体翻译风格（当前测试集多为经文/书名题名），因此可出现“真实译文 PPL 很高”。
3. 单一自动指标不足以给出可靠结论，必须使用多指标联合解释。

### 6.7 补充评估设计（前两项已实现）

为提高论文结论稳健性，我们已实现以下两项补充评估：

1. 双锚点评估（已实现）
- 锚点 A：model-vs-reference（原始评测值）。
- 锚点 B：human_reference-vs-reference（真实译文基线）。
- 聚合输出文件：
  [results/comparison_dual_anchor.csv](results/comparison_dual_anchor.csv)
  [results/comparison_dual_anchor.json](results/comparison_dual_anchor.json)

2. 归一化汇报（已实现）
- 对“越高越好”指标（Lexical/chrF/LLM）：**model_score/human_reference_score**
- 对“越低越好”指标（PPL）：**human_reference_ppl/model_ppl**
- 对应列名：relative_lex, relative_chrf, relative_llm_semantic, relative_llm_fluency, relative_ppl。
- 解释规则：
  1) 取值为 1 表示与 human_reference 持平；
  2) 对 relative_ppl 而言，值越大表示在该 PPL 评估器下越“语言模型友好”；
  3) relative_ppl 普遍大于 1 并不等价于“翻译质量超过人工”，因为当前语料以题名体为主，PPL 评估器与文体存在失配。

尚未实现（后续可扩展）：

1. 语料扩展后再做分层评估：当前样本类型较单一（主要为书名/经文标题），暂不具备稳定的多类型分层前提。后续若引入正文句、注疏句、叙述句等，再进行分层统计。
2. 偏好数据质检：在 DPO 构造阶段强制剔除 non-finite reward 与 chosen=rejected 的样本，记录剔除率。
3. 评测一致性校验：对 LLM Judge 进行重复采样（不同随机种子或多次请求）并报告方差区间。

## 7. 局限性

1. 当前研究仍依赖字典覆盖率，无法完全替代真实双语语义对齐。
2. LLM Judge 存在评测方偏差，结论应与硬指标联合解读。
3. 向量投影替换本质是启发式近似，不保证词法历史学意义上的严格正确。
