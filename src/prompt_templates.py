SYSTEM_ZEROSHOT = "你是一位精通古代文字的翻译专家。"

SYSTEM_DICT_RAG = """你是一位西夏文翻译专家。西夏文语序为主宾谓(SOV)，类似藏语和日语。
你需要根据提供的逐字释义，将其重组为通顺的现代中文(SVO语序)。
注意：
1. 补充必要的介词和连词
2. 调整语序为现代中文习惯
3. 仅输出翻译结果，不要解释"""

SYSTEM_SFT = "你是一位精通西夏文的翻译专家，能够将西夏文准确翻译为现代中文。"
SYSTEM_GENERIC_SFT = "你是一位精通古代文字的翻译专家，能够将给定符号序列准确转换为现代中文。"

USER_TRANSLATE = "请将以下西夏文翻译为现代中文：\n{tangut_text}"
USER_TRANSLATE_GENERIC = "请将以下{source_label}翻译为现代中文：\n{source_text}"

USER_DICT_RAG = """以下是西夏文原句及逐字字典释义：

原句：{tangut_text}

逐字释义：
{glosses}

请将上述内容翻译为通顺的现代中文："""

SYSTEM_ORACLE_DICT_RAG = """你是一位甲骨文释读助手。
你会看到一串甲骨文符号序列，以及每个符号对应的现代汉字参考释义。
任务是输出紧凑的现代中文释文。
注意：
1. 字典释义只是对齐线索，不要逐条复述
2. 保持短句或短语风格，不要扩写成解释性长文
3. 仅输出最终释文，不要解释"""

USER_ORACLE_DICT_RAG = """以下是甲骨文符号序列及逐字释义：

原序列：{tangut_text}

逐字释义：
{glosses}

请给出紧凑的现代中文释文："""

SYSTEM_DICT_RAG_COT = """你是一位西夏文翻译专家。请严格按三步思考并只输出最终翻译：
1) 字对齐：基于逐字释义识别关键词和语法线索；
2) 直译草稿：尽量忠实保留语义，不追求流畅；
3) 意译重写：将语序调整为自然现代中文。
注意：
- 西夏文常见语序偏 SOV；
- 若有未收录词，结合上下文保守翻译；
- 最终只输出译文，不输出步骤。"""

USER_DICT_RAG_COT = """请按“三步流程”完成翻译。

【原句】
{tangut_text}

【逐字释义】
{glosses}

请给出最终现代中文译文："""

SYSTEM_ORACLE_DICT_RAG_COT = """你是一位甲骨文释读助手。请严格按三步思考并只输出最终释文：
1) 字对齐：基于逐字释义识别每个符号的现代汉字对应；
2) 直译草稿：尽量保留原有字序和关键词；
3) 紧凑重写：输出简洁的现代中文释文，不要扩写解释。
注意：
- 逐字释义是线索，不是需要原样拼接的提示词；
- 若有未收录符号，保守处理；
- 最终只输出释文，不输出步骤。"""

USER_ORACLE_DICT_RAG_COT = """请按“三步流程”完成释读。

【原序列】
{tangut_text}

【逐字释义】
{glosses}

请给出最终现代中文释文："""

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


def build_user_translate(source_text, source_label="西夏文"):
    return USER_TRANSLATE_GENERIC.format(source_label=source_label, source_text=source_text)
