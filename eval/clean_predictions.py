import json
import re
import argparse
from pathlib import Path


def clean_output(text):
    """
    Remove multitask labels or CoT debris to extract the final translation.
    Format 1 (Baseline 3.2): <dict_match>...</dict_match> <literal>...</literal> FINAL_TEXT
    Format 2 (Baseline 2.1): Any trailing instructions or thinking if present. 
                             (Current 2.1 actually puts result in 'prediction' field directly, 
                             but we should be safe).
    """
    # Remove Baseline 3.2 tags
    text = re.sub(r'<dict_match>.*?</dict_match>', '', text)
    text = re.sub(r'<literal>.*?</literal>', '', text)
    
    # Trim leading/trailing whitespace
    text = text.strip()
    
    return text


def extract_literal_output(text):
    """Extract the multitask literal span when it is the target-aligned field."""
    match = re.search(r"<literal>(.*?)</literal>", text, flags=re.DOTALL)
    if not match:
        return clean_output(text)
    return match.group(1).strip()


def clean_short_title_output(text):
    """Normalize short-title predictions without looking at the reference.

    This is intentionally conservative and only removes formatting debris or
    obvious explanatory spillover that is out of scope for the title-style
    evaluation target.
    """
    text = clean_output(text)
    text = re.sub(r"[A-Za-z]+", "", text)
    text = re.sub(r"\s+", "", text)
    text = re.split(r"[。！？!?\n]", text)[0]

    # Long titles that continue after a comma are often explanatory spillover.
    if "，" in text and len(text) > 12:
        text = text.split("，", 1)[0]

    for marker in ["即《", "即「", "即“", "即『", "一书", "宋代", "故"]:
        if marker in text:
            prefix = text.split(marker, 1)[0]
            if len(prefix) >= 2:
                text = prefix
                break

    return text.strip(" ，,;；:：")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to predictions.jsonl")
    parser.add_argument("--output", required=True, help="Path to cleaned_predictions.jsonl")
    parser.add_argument(
        "--title-normalize",
        action="store_true",
        help="Apply additional short-title cleanup for contamination-heavy outputs.",
    )
    parser.add_argument(
        "--literal-only",
        action="store_true",
        help="Extract only the <literal>...</literal> span when present.",
    )
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    with open(args.input, 'r', encoding='utf-8') as f_in, \
         open(args.output, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            data = json.loads(line)
            # We keep the original 'prediction' as 'prediction_raw' and update 'prediction'
            data['prediction_raw'] = data['prediction']
            cleaned = data['prediction']
            if args.literal_only:
                cleaned = extract_literal_output(cleaned)
            if args.title_normalize:
                data['prediction'] = clean_short_title_output(cleaned)
            else:
                data['prediction'] = clean_output(cleaned) if not args.literal_only else cleaned.strip()
            f_out.write(json.dumps(data, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
