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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to predictions.jsonl")
    parser.add_argument("--output", required=True, help="Path to cleaned_predictions.jsonl")
    args = parser.parse_args()
    
    with open(args.input, 'r', encoding='utf-8') as f_in, \
         open(args.output, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            data = json.loads(line)
            # We keep the original 'prediction' as 'prediction_raw' and update 'prediction'
            data['prediction_raw'] = data['prediction']
            data['prediction'] = clean_output(data['prediction'])
            f_out.write(json.dumps(data, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
