import json
import re
import jieba
from collections import defaultdict

class TangutDictionaryRebuilder:
    def __init__(self, raw_dict_path: str):
        self.raw_dict_path = raw_dict_path
        
        # 预编译正则表达式：移除所有括号及其中间的内容 (例如 【雅】, （动）)
        self.bracket_pattern = re.compile(r'[【\[（\(].*?[】\]）\)]')
        # 移除标点符号
        self.punctuation_pattern = re.compile(r'[^\w\s]')

    def clean_and_cut_explanation(self, text: str) -> list:
        """清洗噪音并进行中文分词"""
        if not text:
            return []
            
        # 1. 剔除元数据标记
        cleaned_text = self.bracket_pattern.sub('', text)
        # 2. 剔除标点
        cleaned_text = self.punctuation_pattern.sub('', cleaned_text)
        cleaned_text = cleaned_text.strip()
        
        if not cleaned_text:
            return []
            
        # 3. 使用 jieba 切分为有意义的词素，而不是死板的单字
        # 例如："嘬食" -> ["嘬", "食"], "恶毒" -> ["恶毒"]
        words = list(jieba.cut(cleaned_text))
        
        # 将词和拆解后的单字都加入候选集（扩大宽容度）
        final_semantics = set(words)
        for word in words:
            if len(word) > 1:
                final_semantics.update(list(word))
                
        return list(final_semantics)

    def rebuild(self, output_path: str):
        with open(self.raw_dict_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
            
        # 按照西夏文的长度分类存储，方便后续做“最大前向匹配”
        # 结构: { "1": { "𘂦": ["割", "折", "割折"] }, "2": { "𗀀𗻊": ["毒", "毒药"] } }
        processed_dict = defaultdict(dict)
        
        for entry in raw_data:
            # 提取西夏文主键
            tangut_key = entry.get('word') or entry.get('character')
            if not tangut_key:
                continue
                
            raw_cn = entry.get('explanationCN', '')
            
            # 清洗并切分出语义集合
            semantic_set = self.clean_and_cut_explanation(raw_cn)
            
            if semantic_set:
                length_category = str(len(tangut_key))
                
                # 处理多重释义合并（如果有重复录入的条目）
                if tangut_key in processed_dict[length_category]:
                    existing_set = set(processed_dict[length_category][tangut_key])
                    existing_set.update(semantic_set)
                    processed_dict[length_category][tangut_key] = list(existing_set)
                else:
                    processed_dict[length_category][tangut_key] = semantic_set
                    
        # 写入大模型专用的高密度检索字典
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(processed_dict, f, ensure_ascii=False, indent=2)
            
        print(f"字典重构完毕！已剔除噪音并完成 jieba 语义切分，输出至: {output_path}")

# 运行重构
if __name__ == "__main__":
    rebuilder = TangutDictionaryRebuilder("dictionary.json")
    rebuilder.rebuild("reward_dict.json")