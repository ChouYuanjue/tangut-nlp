SYSTEM_ZEROSHOT = "你是一位精通古代文字的翻译专家。"

SYSTEM_DICT_RAG = """你是一位西夏文翻译专家。西夏文语序为主宾谓(SOV)，类似藏语和日语。
你需要根据提供的逐字释义，将其重组为通顺的现代中文(SVO语序)。
注意：
1. 补充必要的介词和连词
2. 调整语序为现代中文习惯
3. 仅输出翻译结果，不要解释"""

SYSTEM_SFT = "你是一位精通西夏文的翻译专家，能够将西夏文准确翻译为现代中文。"

USER_TRANSLATE = "请将以下西夏文翻译为现代中文：\n{tangut_text}"

USER_DICT_RAG = """以下是西夏文原句及逐字字典释义：

原句：{tangut_text}

逐字释义：
{glosses}

请将上述内容翻译为通顺的现代中文："""

JUDGE_TEMPLATE = """你是一位西夏文研究专家和翻译评审员。请评估以下翻译的质量。

## 西夏文原文
{tangut_input}

## 逐字字典释义
{dictionary_glosses}

## 候选翻译
{candidate_translation}

请从以下两个维度打分（1-5分）：

1. **语义完整性** (1-5)：翻译是否包含了原文所有关键语义？
2. **流畅度** (1-5)：中文是否通顺自然？

请严格按以下JSON格式输出：
{{"semantic_completeness": <int>, "fluency": <int>, "reasoning": "<brief explanation>"}}"""


def build_chat_prompt(system, user):
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def build_sft_sample(system, user, assistant):
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n{assistant}<|im_end|>"
    )
