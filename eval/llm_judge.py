"""
Metric 4: LLM-as-a-Judge (mock for now)
Uses an LLM to rate translation quality on semantic completeness and fluency.
Currently provides a mock implementation with random scores.
"""

import json
import random
import sys
from pathlib import Path
from typing import List, Optional
import time

from openai import AzureOpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.azure_openai_config import (
    DEFAULT_AZURE_OPENAI_API_VERSION,
    resolve_azure_openai_config,
)

class LLMJudgeScorer:
    """LLM-based judge for translation quality assessment.

    Uses an Azure OpenAI API endpoint to evaluate translations.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        deployment: Optional[str] = None,
        api_version: str = DEFAULT_AZURE_OPENAI_API_VERSION,
        mock: bool = False,
        timeout: int = 30,
    ):
        """Initialize the LLM judge.

        Args:
            api_key: API key for the LLM service.
            endpoint: Azure OpenAI endpoint override.
            deployment: Azure OpenAI deployment override.
            api_version: Azure OpenAI API version.
            mock: If True, return random scores instead of calling an LLM.
            timeout: Timeout in seconds for API calls.
        """
        self.mock = mock
        self.timeout = timeout
        self.api_version = api_version

        if self.mock:
            self.api_key = api_key
            self.endpoint = endpoint
            self.deployment = deployment or ""
            self.client = None
            return

        config = resolve_azure_openai_config(
            api_key=api_key,
            endpoint=endpoint,
            deployment=deployment,
        )
        self.api_key = config["api_key"]
        self.endpoint = config["endpoint"]
        self.deployment = config["deployment"]

        self.client = AzureOpenAI(
            azure_endpoint=self.endpoint,
            api_key=self.api_key,
            api_version=self.api_version,
            timeout=timeout,
        )

    def score(
        self,
        tangut_input: str,
        candidate: str,
        dictionary_glosses: list,
        max_retries: int = 2,
    ) -> dict:
        """Score a single translation using the LLM judge.

        Args:
            tangut_input: Source Tangut string.
            candidate: Candidate Chinese translation.
            dictionary_glosses: List of expected dictionary glosses for
                reference context.
            max_retries: Maximum number of retry attempts for API calls.

        Returns:
            Dict with keys:
                - "semantic_completeness": int 1-5
                - "fluency": int 1-5
                - "reasoning": str explanation
        """
        if self.mock:
            return self._mock_score()

        gloss_str = ", ".join(str(g) for g in dictionary_glosses)
        prompt = (
            f"作为一位精通古汉语与西夏文句法结构（主宾谓结构为主）的专家候选人翻译质量评价者，请评估以下翻译。\n\n"
            f"西夏文原文: {tangut_input}\n"
            f"逐字字典释义: {gloss_str}\n"
            f"候选体翻译: {candidate}\n\n"
            f"请根据以下两个维度给出1到5的整数评分：\n"
            f"1. 语义完整度 (semantic_completeness): 是否将释义中的关键信息全都翻译出来了，无遗漏事实或错译。\n"
            f"2. 流畅度 (fluency): 汉语语序（SVO）是否通顺，是否符合汉语的自然表达逻辑。\n"
            f"请严格以JSON格式返回，不需要其他文字说明。格式示例：{{\"semantic_completeness\": 4, \"fluency\": 5, \"reasoning\": \"简要理由\"}}"
        )

        for attempt in range(max_retries):
            try:
                response = self.client.responses.create(
                    model=self.deployment,
                    instructions="You are a helpful AI assistant serving as a judge for text translation. Be concise and format the output as JSON strictly.",
                    input=prompt,
                    max_output_tokens=300,
                )
                
                content = response.output_text
                
                # Helper to strip markdown code blocks if the model includes them
                if content is None:
                    content = "{}"
                else:
                    content = content.strip()
                    if content.startswith("```json"):
                        content = content.replace("```json\n", "", 1)
                    elif content.startswith("```"):
                        content = content.replace("```\n", "", 1)
                        
                    if content.endswith("```"):
                        content = content.rpartition("```")[0]
                    
                    # Find the actual JSON object to avoid trailing parsing errors
                    start_idx = content.find('{')
                    end_idx = content.rfind('}') + 1
                    if start_idx != -1 and end_idx != 0:
                        content = content[start_idx:end_idx]
                    else:
                        content = "{}"
                    
                parsed = json.loads(content)
                
                # Extract values, defaulting if parsing fails slightly
                return {
                    "semantic_completeness": int(parsed.get("semantic_completeness", 3)),
                    "fluency": int(parsed.get("fluency", 3)),
                    "reasoning": str(parsed.get("reasoning", "No reasoning provided.")),
                }
            except (TimeoutError, ConnectionError) as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"  [TIMEOUT] After {max_retries} retries: {e}")
                    return {
                        "semantic_completeness": 3,
                        "fluency": 3,
                        "reasoning": f"Timeout after {max_retries} retries",
                    }
            except Exception as e:
                # Fallback handling just in case of API error or JSON parsing error
                print(f"  [ERROR] {e}")
                return {
                    "semantic_completeness": 3,
                    "fluency": 3,
                    "reasoning": f"Error: {str(e)[:50]}",
                }

    def _mock_score(self) -> dict:
        """Generate random mock scores for testing.

        Returns:
            Dict with random semantic_completeness (1-5), fluency (1-5),
            and a placeholder reasoning string.
        """
        semantic = random.randint(1, 5)
        fluency = random.randint(1, 5)
        return {
            "semantic_completeness": semantic,
            "fluency": fluency,
            "reasoning": (
                f"[MOCK] Semantic completeness: {semantic}/5, "
                f"Fluency: {fluency}/5. "
                "This is a placeholder score for testing purposes."
            ),
        }

    def score_batch(
        self,
        inputs: List[str],
        candidates: List[str],
        glosses_list: List[list],
    ) -> dict:
        """Score a batch of translations.

        Args:
            inputs: List of source Tangut strings.
            candidates: List of candidate Chinese translations.
            glosses_list: List of gloss lists (one per example).

        Returns:
            Dict with keys:
                - "mean_semantic_completeness": float
                - "mean_fluency": float
                - "scores": list of individual score dicts
        """
        scores = []
        total = len(inputs)
        
        for idx, (inp, cand, glosses) in enumerate(zip(inputs, candidates, glosses_list)):
            sys.stdout.write(f"\r  Processing {idx+1}/{total}")
            sys.stdout.flush()
            
            try:
                score = self.score(inp, cand, glosses)
                scores.append(score)
            except Exception as e:
                print(f"\n[WARNING] Score {idx+1} failed: {e}")
                scores.append({
                    "semantic_completeness": 3,
                    "fluency": 3,
                    "reasoning": f"Error: {str(e)[:50]}",
                })
        
        sys.stdout.write(f"\r  Processing {total}/{total} ✓\n")
        sys.stdout.flush()

        if not scores:
            return {
                "mean_semantic_completeness": 0.0,
                "mean_fluency": 0.0,
                "scores": [],
            }

        mean_semantic = sum(s["semantic_completeness"] for s in scores) / len(scores)
        mean_fluency = sum(s["fluency"] for s in scores) / len(scores)

        return {
            "mean_semantic_completeness": mean_semantic,
            "mean_fluency": mean_fluency,
            "scores": scores,
        }
