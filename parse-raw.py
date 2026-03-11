from bs4 import BeautifulSoup
import json
import argparse
from typing import Any, Dict, List


def parse_babelstone_html(html_content: str, output_file: str) -> None:
    """解析 BabelStone 西夏文献总目 HTML，生成 SFT 训练数据。

    保留原有功能以备不时之需；脚本主要通过
    :func:`parse_raw_json` 处理预先准备好的 ``raw.json``。
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # 每一个 <tr> 包含一条完整的文献记录
    rows = soup.find_all('tr')
    
    extracted_data: List[Dict[str, Any]] = []
    
    for row in rows:
        # 提取西夏文
        tangut_tag = row.find('p', class_='tangut')
        # 提取中文
        chinese_tag = row.find('p', class_='bs_han')
        
        if tangut_tag and chinese_tag:
            tangut_text = tangut_tag.get_text(strip=True)
            chinese_text = chinese_tag.get_text(strip=True)
            
            # 可选：提取拟音（Sofronov's phonetic reconstruction），用于元数据
            ipa_tag = row.find('p', class_='ipa')
            ipa_text = ipa_tag.get_text(strip=True) if ipa_tag else ""
            
            sft_sample = {
                "instruction": "请将以下西夏文翻译为现代中文：",
                "input": tangut_text,
                "output": chinese_text,
                "metadata": {"phonetic": ipa_text},
            }
            extracted_data.append(sft_sample)
    _write_jsonl(extracted_data, output_file)


def parse_raw_json(input_file: str, output_file: str) -> None:
    """读取固定的 ``raw.json`` 并生成 JSONL 格式 SFT 数据。

    假定 ``raw.json`` 是一个 JSON 数组，每个元素包含
    至少 ``tangut`` 和 ``chinese`` 两个字段，额外可带 ``ipa``。
    当结构不同时请自行调整字段名称。
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    extracted_data: List[Dict[str, Any]] = []

    if not isinstance(data, list):
        raise ValueError(f"预期 ``raw.json`` 为数组, 但得到 {type(data)}")

    for entry in data:
        # 支持多个可能的 key 名称
        tangut_text = entry.get('tangut') or entry.get('tangut_text') or ''
        chinese_text = entry.get('chinese') or entry.get('bs_han') or ''
        ipa_text = entry.get('ipa', '')

        if tangut_text and chinese_text:
            sft_sample = {
                "instruction": "请将以下西夏文翻译为现代中文：",
                "input": tangut_text,
                "output": chinese_text,
                "metadata": {"phonetic": ipa_text},
            }
            extracted_data.append(sft_sample)

    _write_jsonl(extracted_data, output_file)


def _write_jsonl(items: List[Dict[str, Any]], output_file: str) -> None:
    """将列表写入 jsonl，并打印基本信息和两条预览。"""
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"解析完成！共成功提取 {len(items)} 条高质量平行序列，已保存至 {output_file}。")
    for i in range(min(2, len(items))):
        print(f"预览 {i+1}: {items[i]}")


def main():
    parser = argparse.ArgumentParser(
        description="生成 SFT 数据：从 raw.json 或 BabelStone HTML 读取。"
    )
    parser.add_argument(
        "--input",
        help="输入文件（默认 raw.json）",
        default="raw.json",
    )
    parser.add_argument(
        "--output",
        help="输出 jsonl 文件（默认 babelstone_baseline_sft.jsonl）",
        default="babelstone_baseline_sft.jsonl",
    )
    parser.add_argument(
        "--mode",
        choices=["json", "html"],
        default="json",
        help="解析模式：json 或 html。",
    )
    args = parser.parse_args()

    if args.mode == 'json':
        parse_raw_json(args.input, args.output)
    else:
        with open(args.input, 'r', encoding='utf-8') as f:
            html = f.read()
        parse_babelstone_html(html, args.output)


if __name__ == '__main__':
    main()

# 测试运行（直接将你提供的源码片段作为输入）
sample_html = """
[此处粘贴你提供的 HTML 源码]
"""

# 取消注释以运行
# parse_babelstone_html(sample_html, "babelstone_baseline_sft.jsonl")