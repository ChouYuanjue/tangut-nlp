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

### 2.1 推理层过渡（Inference Track）

1. Baseline 1: Zero-shot
- 直接对西夏文输入进行指令翻译。
- 目标：估计大模型对西夏文 Unicode 的原生可迁移能力。

2. Baseline 2: Dictionary-Augmented RAG
- 以字典释义作为外部知识注入，不更新模型参数。
- 目标：测量“仅靠符号注入”的上限。

3. Baseline 2.1: RAG + 三步 CoT
- Prompt 显式拆分为“字面对齐-直译-意译重写”。
- 变量：仅改变推理提示词结构。

### 2.2 数据与 SFT 过渡（Data & SFT Track）

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

### 2.3 架构与对齐过渡（Architecture & Alignment Track）

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

在 DPO 候选构建中，使用如下奖励形式进行排序：

$$
R(y)=\alpha\cdot\text{LexicalCoverage}(y)-\beta\cdot\text{PPL}(y)
$$

其中 $y$ 为候选翻译，最高分样本作为 Chosen，最低分样本作为 Rejected。

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
- 各基线指标：results/<experiment>/metrics.json
- 总表：results/comparison.csv 与 results/comparison.json
- 可视化：results/comparison.png 与 results/comparison_radar.png

## 6. 现有结果分析（忠实记录）

本节基于当前仓库中的已落盘结果进行陈述，数据来源为 [results/comparison.csv](results/comparison.csv) 与 [results/comparison.json](results/comparison.json)。

图表引用：

1. 总览柱状图：[results/comparison.png](results/comparison.png)
2. 全方法雷达图：[results/comparison_radar.png](results/comparison_radar.png)
3. 分方法雷达图：[results/comparison_radar_each.png](results/comparison_radar_each.png)

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

### 6.3 忠实记录表（当前快照，n=50）

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

注：该表仅陈述当前仓库快照，不对未来重训后的变化作外推。

## 7. 局限性

1. 当前研究仍依赖字典覆盖率，无法完全替代真实双语语义对齐。
2. LLM Judge 存在评测方偏差，结论应与硬指标联合解读。
3. 向量投影替换本质是启发式近似，不保证词法历史学意义上的严格正确。
