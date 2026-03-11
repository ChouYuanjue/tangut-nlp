import json
import re
from collections import defaultdict


class OptimizedDictEntry:
    __slots__ = ('key', 'GX', 'GHC', 'LFW', 'explanationEN', 'explanationCN', 'entry_type')

    def __init__(self, data):
        if "word" in data:
            self.key = data["word"]
            self.entry_type = "word"
        elif "character" in data:
            self.key = data["character"]
            self.entry_type = "character"
        else:
            self.key = ""
            self.entry_type = "unknown"
        self.GX = data.get("GX", "")
        self.GHC = data.get("GHC", "")
        self.LFW = data.get("LFW", "")
        self.explanationEN = data.get("explanationEN", "")
        self.explanationCN = data.get("explanationCN", "")


class BilingualDictionary:
    def __init__(self, dict_file_path):
        self.forward_index = {}
        self.reverse_index = defaultdict(list)
        with open(dict_file_path, 'r', encoding='utf-8') as f:
            dict_data = json.load(f)
            self._build_indexes(dict_data)

    def _build_indexes(self, dict_data):
        for item in dict_data:
            entry = OptimizedDictEntry(item)
            self.forward_index[entry.key] = entry
            if entry.explanationCN:
                cn_keywords = [kw.strip() for kw in re.findall(r'[^，。！？；：]+', entry.explanationCN) if kw.strip()]
                for kw in cn_keywords:
                    if not re.match(r'^[0-9【】]+$', kw):
                        self.reverse_index[kw].append(entry.key)
            if entry.explanationEN:
                en_keywords = [kw.strip().lower() for kw in re.split(r'[\s.,;]+', entry.explanationEN) if kw.strip()]
                for kw in en_keywords:
                    self.reverse_index[kw].append(entry.key)

    def search_by_key(self, key):
        return self.forward_index.get(key)

    def get_glosses(self, tangut_text):
        """Maximum forward matching over tangut_text. Returns list of (substr, entry_or_None)."""
        results = []
        i = 0
        while i < len(tangut_text):
            matched = False
            for length in range(min(5, len(tangut_text) - i), 0, -1):
                substr = tangut_text[i:i+length]
                entry = self.forward_index.get(substr)
                if entry:
                    results.append((substr, entry))
                    i += length
                    matched = True
                    break
            if not matched:
                results.append((tangut_text[i], None))
                i += 1
        return results

    def get_stats(self):
        word_count = sum(1 for e in self.forward_index.values() if e.entry_type == "word")
        char_count = sum(1 for e in self.forward_index.values() if e.entry_type == "character")
        return {
            "total_entries": len(self.forward_index),
            "words": word_count,
            "characters": char_count,
            "keywords": len(self.reverse_index),
        }
