# Structured Synthetic Supervision and Metric Pitfalls in Ultra-Low-Resource Tangut Short-Text Translation

## Abstract

Tangut translation is an extreme low-resource problem: in our repository snapshot, only 491 real Tangut-Chinese pairs are available, with 400/41/50 train-dev-test examples. The task is also structurally unusual, because many targets are short, title-like Chinese strings rather than ordinary modern sentences. We evaluate three families of methods in this setting: inference-only prompting, synthetic-data supervised fine-tuning (SFT), and direct preference optimization (DPO). The strongest current result comes from a structured multitask SFT design that exposes intermediate alignment structure during training; it achieves the best non-reference chrF++ score (25.99) and the highest exact-match count (5/50) among all learned systems in the repository snapshot. We further show that several apparently reasonable reference-free metrics are misaligned with the task: the gold human reference itself receives only moderate dictionary coverage and extremely poor perplexity under a modern-Chinese language model, despite achieving perfect chrF++. Finally, we report a negative result for reward-based alignment: DPO trained with a lexical-coverage-plus-perplexity reward degrades reference-aligned performance and introduces output contamination. These findings suggest that, for ultra-low-resource Tangut short-text translation, explicit structural supervision is currently more reliable than preference optimization, and evaluation must be anchored to reference-aware metrics rather than generic fluency proxies.

## 1. Introduction

Machine translation for historical and ultra-low-resource languages is difficult for two separate reasons. First, parallel data is often scarce or nearly absent. Second, the outputs that matter to scholars are not always long, natural modern sentences: they may be titles, glosses, liturgical fragments, or compact bibliographic strings. Tangut is a canonical example. The script is historically important, but usable parallel Tangut-Chinese resources remain tiny. In the current repository snapshot, we only have 491 real Tangut-Chinese pairs.

This small-data regime creates a practical question: when full supervised translation is impossible, what kind of signal is most helpful? A natural first option is dictionary-augmented prompting. A second option is to synthesize pseudo-parallel training data from dictionary knowledge and monolingual Chinese. A third option is to add preference optimization on top of an SFT base. The repository already contains experiments spanning all three options, which makes it possible to write a focused empirical paper from the existing evidence.

Our main finding is that **structured synthetic supervision currently works best**. Among all non-reference systems, the strongest model is a multitask SFT variant that asks the model to expose intermediate alignment structure during training. This system is not best because it maximizes dictionary coverage or because it looks best under a generic Chinese language model. It is best because it most closely matches the reference outputs in shape and content: it has the highest current chrF++, the highest exact-match rate, and the best title-suffix preservation among learned systems.

The repository also reveals two important failure modes. First, **reference-free metrics can be badly misleading**. The human reference itself is penalized by lexical coverage and perplexity, which means those metrics cannot serve as primary model-selection criteria for this task. Second, **preference optimization is unstable under the current reward design**. The DPO model underperforms its SFT base and frequently produces contaminated outputs, suggesting a mismatch between the reward and the actual task objective.

These observations lead to a paper with three contributions:

1. an empirical comparison of prompting, synthetic SFT, and DPO for Tangut short-text translation;
2. a positive method result showing that structured synthetic supervision is the strongest current approach;
3. an evaluation and alignment analysis showing why reference-free metrics and the current DPO reward can fail.

## 2. Task Setting

The real Tangut data in this repository contains 491 pairs. The current split is 400 training examples, 41 development examples, and 50 test examples. Unlike ordinary machine-translation benchmarks, the targets are short and often title-like. Many examples resemble book titles, sutra titles, or compact scholastic expressions. This matters because title reconstruction rewards precise lexical choices and compact formatting, while a modern Chinese fluency model may incorrectly treat those outputs as unnatural.

Because the real dataset is tiny, the system uses synthetic data to train translation models. For each SFT branch, 50,000 synthetic Tangut-Chinese pairs are produced from monolingual classical Chinese and dictionary knowledge, then combined with the 400 real training examples to form a 54,000-example training set. All learned models use the same instruction-tuned 7B backbone.

We evaluate with two classes of metrics. The repository's existing metrics are lexical coverage, perplexity, chrF++, and a dictionary-conditioned LLM judge. In this paper draft, we argue that these metrics should not be treated equally. chrF++ is the most trustworthy current primary metric because it correctly gives the human reference a perfect score. Lexical coverage, perplexity, and the current LLM judge are better interpreted as diagnostic signals.

## 3. Methods

### 3.1 Inference-only baselines

The first family contains no weight updates. `baseline1` directly prompts the base model with the Tangut string. `baseline2` augments the prompt with dictionary glosses. `baseline2_1_cot` further restructures the prompt into a three-step process intended to encourage alignment, literal drafting, and rewriting.

These methods are important because they show what can be achieved from symbol injection alone. In the current results, dictionary prompting is substantially better than zero-shot generation, but it often expands compact titles into explanatory paraphrases rather than reconstructing the reference form.

### 3.2 Synthetic SFT baselines

The second family uses synthetic training data.

- `baseline3` uses mixed Tangut-Chinese inputs.
- `baseline3_1_unk` enforces pure Tangut inputs by replacing dictionary misses with `UNK`.
- `baseline3_3_semantic` also enforces pure inputs, but replaces misses through vector-space semantic projection instead of `UNK`.
- `baseline3_2_multitask` introduces a structured target containing explicit alignment scaffolding.

The central design question is whether the model benefits more from input purity, approximate semantic preservation, or explicit alignment structure. The current evidence favors explicit alignment structure.

### 3.3 Preference optimization

The final family applies DPO to an SFT base using reward-defined chosen/rejected pairs. In the current repository, the reward combines lexical coverage with a penalty on log perplexity. The pipeline first selects the best SFT base among the mixed, `UNK`, and semantic variants using chrF++, then builds preference pairs from that base. In the current snapshot, the selected base is the `UNK` model.

This design is informative even though it does not currently succeed. Because the reward is partly built from misaligned signals, DPO becomes a useful negative-result case study.

## 4. Main Results

Table 1 summarizes the repository metrics. The central pattern is straightforward. Zero-shot generation fails. Dictionary prompting is much stronger, especially on lexical coverage. Synthetic SFT improves reference alignment substantially. Among the SFT family, the multitask structured variant is best. DPO does not improve on its SFT base.

The most important comparison is between `baseline3_1_unk`, `baseline3_2_multitask`, and `final_v2`. The `UNK` model already demonstrates that pure-input training is helpful. The multitask model improves further, reaching chrF++ 25.99 and 5 exact matches out of 50 test items. By contrast, `final_v2` falls back to chrF++ 19.39 and only 2 exact matches, while also showing visible contamination in some outputs.

This result suggests that, in the current Tangut setting, preference optimization is not the easiest way to exploit scarce supervision. A better strategy is to inject stronger structure during supervised training.

## 5. Analysis

### 5.1 Reference-aligned diagnostics

Repository-level metrics alone do not fully explain why `baseline3_2_multitask` is the best current model. Additional diagnostics from stored predictions make the result clearer. The multitask model produces the best exact-match count (5/50), the best title-suffix preservation among learned systems (19/27), and a near-reference output length ratio (1.10). These properties matter for a title-like task.

By contrast, the dictionary-prompting baseline often produces outputs that are too long and too explanatory. Its average output is about 1.83 times the reference length. This helps lexical coverage but hurts exact title reconstruction.

### 5.2 Human-reference anchor

One of the strongest pieces of evidence in the repository is the `human_reference` experiment. The gold reference receives perfect chrF++, as expected, but only moderate lexical coverage and catastrophic perplexity. This reveals a core evaluation issue: the current lexical coverage metric measures overlap with dictionary-derived semantics rather than agreement with the gold title, while the perplexity model penalizes compact title style and historical transliterations.

The implication is methodological, not cosmetic. If we were to select models purely by lexical coverage or perplexity, we would risk preferring systems that paraphrase or overgenerate instead of systems that best reconstruct the true title. Therefore, in the paper, lexical coverage and perplexity should be reported as supporting diagnostics, not as the primary decision criteria.

### 5.3 Why the current DPO fails

The DPO result is not a random bad run. The preference data itself is imperfect. The repository contains 3,867 preference pairs, including a small number of duplicate or invalid entries. More importantly, a quick audit against the original synthetic targets shows that the reward-chosen candidate is closer to gold only about 71% of the time under a simple similarity proxy, and the low-gap pairs are much noisier. This means that the reward is informative but far from clean.

The current selector also excludes the strongest multitask model from the DPO-base search. As a result, the paper should avoid claiming that "DPO does not work for Tangut" in general. A narrower and defensible statement is: **under the current reward design and current base-selection policy, DPO is unstable and underperforms structured SFT**.

## 6. Discussion and Limitations

This paper draft should be explicit about scope. The current dataset supports Tangut short-text translation, especially title-like strings, not general open-domain Tangut sentence translation. That scope is still meaningful: many historically important Tangut materials appear precisely in compact catalog, title, or scholastic forms, and low-resource methods for such data are valuable.

The paper should also be explicit about what remains unfinished. The related-work section still needs verified citations. A reviewer-facing follow-up experiment should test either a stricter DPO pair filter or a multitask-base DPO variant. A reference-aware judge would also strengthen the evaluation section.

## 7. Conclusion

The current repository already supports a coherent paper. Structured multitask synthetic supervision is the strongest method in the snapshot. Pure-input training is helpful, but noisy semantic projection does not yet outperform the simpler `UNK` strategy. Several reference-free metrics are misaligned enough to mis-rank the human reference. DPO, under the current reward and base-selection design, is unstable and best treated as a negative result. Together, these findings argue for a simple practical lesson: in ultra-low-resource Tangut short-text translation, **explicit structural supervision is currently more reliable than reward-based alignment, and evaluation must stay anchored to reference-aware evidence.**

## Limitations

- The current paper uses a very small real Tangut dataset.
- The task is short-text and title-like, so conclusions do not automatically transfer to long-form sentence translation.
- The current DPO conclusion is limited to the reward design and base-selection policy used here.
- The related-work section still needs verified bibliography.
