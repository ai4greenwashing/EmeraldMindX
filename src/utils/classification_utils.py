from __future__ import annotations

import os
import json
from itertools import cycle
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import pandas as pd
from neo4j import GraphDatabase, Result
from graphdatascience import GraphDataScience
from sentence_transformers import SentenceTransformer
from google import genai
import re, unicodedata
from rapidfuzz import fuzz, process, utils
from numpy import dot
from numpy.linalg import norm
import numpy as np
from dotenv import load_dotenv
from google.genai import types
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from typing import List, Dict, Any, Tuple
import json
load_dotenv()
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from typing import List, Dict, Any, Tuple
import json
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
import os
import json
from itertools import cycle
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import pandas as pd
from neo4j import GraphDatabase, Result
from graphdatascience import GraphDataScience
from sentence_transformers import SentenceTransformer
from google import genai
import re, unicodedata
from rapidfuzz import fuzz, process, utils
from numpy import dot
from numpy.linalg import norm
import numpy as np
from dotenv import load_dotenv
from google.genai import types
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from typing import List, Dict, Any, Tuple
import json
load_dotenv()
try:
    driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "emeraldmind"))
    driver.verify_connectivity()
    gds = GraphDataScience(driver)
except Exception as e:
    raise RuntimeError(f"Failed to connect to Neo4j: {e}")
encoder = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")
# client1 = genai.Client(api_key=os.getenv("GEMINI_API_KEY_1"))
# client2 = genai.Client(api_key=os.getenv("GEMINI_API_KEY_2"))
# client3 = genai.Client(api_key=os.getenv("GEMINI_API_KEY_3"))
# client4 = genai.Client(api_key=os.getenv("GEMINI_API_KEY_4"))
# client5 = genai.Client(api_key=os.getenv("GEMINI_API_KEY_5"))
# client6 = genai.Client(api_key=os.getenv("GEMINI_API_KEY_6"))
# client7 = genai.Client(api_key=os.getenv("GEMINI_API_KEY_7"))
# client8 = genai.Client(api_key=os.getenv("GEMINI_API_KEY_8"))
# cycle_clients = cycle([client1, client2, client3, client4, client5, client6, client7, client8])

MODEL = "google/gemma-3-27b-it"

# variable used in names
MODEL_NAME="gemma3-27b" 

processor = AutoProcessor.from_pretrained(MODEL)
model = Gemma3ForConditionalGeneration.from_pretrained(
    MODEL,
    device_map="auto",
    torch_dtype=torch.bfloat16
).eval()



# Classes that should always be included using the full claim embedding
ALWAYS_INCLUDE_CLASSES = ["KPIObservation", "Penalty"]

ALIASES = {
    "Organization": "org",
    "Person": "per",
    "Facility": "fac",
    "Product": "prod",
    "Material": "mat",
    "Emission": "emi",
    "Waste": "wst",
    "KPIObservation": "kpi",
    "Standard": "std",
    "Certification": "cert",
    "Regulation": "reg",
    "Initiative": "init",
    "Goal": "goal",
    "Sustainabilityclaim": "sc",
    "ThirdPartyVerification": "tpv",
    "CarbonOffsetProject": "cop",
    "ScienceBasedTarget": "sbt",
    "Controversy": "cont",
    "Penalty": "pen",
    "MediaReport": "med",
    "Community": "comm",
    "Location": "loc",
    "Authority": "auth",
    "Country": "cntr",
    "claimKeyword": "ckey",
    "Investment": "inv",
    "Project": "proj",
}

_CORP_SUFFIXES = {
    "inc",
    "incorporated",
    "corp",
    "corporation",
    "company",
    "co",
    "ltd",
    "limited",
    "plc",
    "sa",
    "ag",
    "bv",
    "oy",
}

FEW_TMPL=""""
You are an **ESG (Environmental, Social, and Governance) and Greenwashing Fact-Checker.**
 
**Task:** Given an ESG-related claim, fact-check it and determine whether it constitutes **greenwashing**, **not_greenwashing**, or if you should **abstain**.
 
### Fact-Checking Process
 
**When context is provided below:** Use the provided context as your PRIMARY source of truth. Cross-reference claims against specific information in the context.
 
**When no context is provided:** Use your internal knowledge to verify factual accuracy. You may only make determinations based on facts you can confidently recall.
 
**In BOTH cases:** Your decision MUST be based on verifiable facts that either prove or disprove the claim. NEVER judge based on:
- Vague wording or marketing language
- How "suspicious" or "too good to be true" something sounds  
- Absence of detail (this warrants abstain, not greenwashing)
- Your intuition or reasoning about what "could be" misleading
 
### Decision Criteria
 
**Greenwashing** - You have specific evidence that the claim is factually false or misleading:
1. **Type 1 - Factually false labels:** You can verify the label/certification is false, doesn't exist, or misrepresents actual standards.
2. **Type 2 - Legal obligations misrepresented as voluntary:** You can verify the action was legally mandated, not a voluntary initiative.
3. **Type 3 - Partial truth presented as whole:** You can verify only part of the product/service meets the stated criteria.
4. **Type 4 - Claims contradicted by evidence:** You have specific facts/evidence that directly contradict the claim.
 
**Not Greenwashing** - You can verify the claim is factually accurate based on available information (context or knowledge).
 
**Abstain** - You cannot verify factual accuracy either way. Use this when:
- No context is provided AND you lack sufficient internal knowledge
- Context is provided but doesn't contain information to verify/refute the specific claim
- You cannot confidently determine if the claim is true or false
 
### FEW-SHOT EXAMPLES
 
<EXAMPLE 1: No context provided, insufficient internal knowledge>
<CONTEXT>
[No context provided]
</CONTEXT>
<CLAIM>H&M promotes its new "Conscious Collection" as a sustainable fashion line.</CLAIM>
<OUTPUT>
Label: abstain
Type: N/A
Justification: No context provided. While I know H&M has a "Conscious Collection," I do not have specific, verifiable information about the materials, certifications, or manufacturing processes used in this collection to determine if the sustainability claims are factually accurate or false.
</OUTPUT>
</EXAMPLE 1>
---
<EXAMPLE 2: No context provided, using internal knowledge>
<CONTEXT>
[No context provided]
</CONTEXT>
<CLAIM>McDonald's announces it is reducing plastic waste by introducing reusable cutlery in France.</CLAIM>
<OUTPUT>
Label: greenwashing
Type: Type 2
Justification: Based on internal knowledge: The French Anti-Waste Law (AGEC), passed in February 2020, mandates that fast-food restaurants must use reusable tableware for on-site dining as of January 1, 2023. McDonald's is legally required to provide reusable cutlery in France, making this a legal compliance issue rather than a voluntary sustainability initiative.
</OUTPUT>
</EXAMPLE 2>
---
<EXAMPLE 3: Context provided, contradicts claim>
<CONTEXT>
Product manufacturing report: "Walmart's bamboo towel line uses bamboo viscose, produced through a chemical process involving carbon disulfide (CS2) and sodium hydroxide (NaOH). The manufacturing process generates significant chemical waste and requires substantial water usage."
</CONTEXT>
<CLAIM>Walmart advertises its "eco-friendly" bamboo towels as fully sustainable.</CLAIM>
<OUTPUT>
Label: greenwashing
Type: Type 3
Justification: The provided context shows that while bamboo is the raw material, the manufacturing process involves "carbon disulfide (CS2) and sodium hydroxide (NaOH)" and "generates significant chemical waste and requires substantial water usage." The claim of "fully sustainable" is factually inaccurate as it ignores the environmentally harmful manufacturing process documented in the context.
</OUTPUT>
</EXAMPLE 3>
---
<EXAMPLE 4: Context provided, supports claim with evidence>
<CONTEXT>
Regulatory filing: "Canada's Competition Bureau issued a penalty to Keurig Canada in January 2022 for making false or misleading claims about the recyclability of its K-Cup pods. Investigation found that the majority of Canadian municipalities do not accept K-Cup pods in curbside recycling programs due to their multi-material composition."
</CONTEXT>
<CLAIM>Keurig states that all of its single-use coffee pods are "easily recyclable in any city recycling program."</CLAIM>
<OUTPUT>
Label: greenwashing
Type: Type 4
Justification: The context directly contradicts the claim, stating that "the majority of Canadian municipalities do not accept K-Cup pods in curbside recycling programs" and that Keurig was penalized by Canada's Competition Bureau in 2022 "for making false or misleading claims about the recyclability" of these pods.
</OUTPUT>
</EXAMPLE 4>
---
<EXAMPLE 5: Context provided, verifies claim>
<CONTEXT>
Energy audit report: "Tesla's Gigafactory 1 in Nevada operates with a 70 MW solar array installed on its roof, supplemented by renewable energy purchased through power purchase agreements (PPAs). As of Q4 2023, 100% of the facility's energy consumption comes from renewable sources, verified by third-party auditor SGS."
</CONTEXT>
<CLAIM>Tesla reports that its Gigafactory in Nevada now runs entirely on renewable energy.</CLAIM>
<OUTPUT>
Label: not_greenwashing
Type: N/A
Justification: The context verifies the claim with specific details: "70 MW solar array" on-site, renewable energy PPAs, and third-party verification by SGS confirming "100% of the facility's energy consumption comes from renewable sources." The claim is factually accurate based on documented evidence.
</OUTPUT>
</EXAMPLE 5>
---
<EXAMPLE 6: Context provided but insufficient>
<CONTEXT>
Press release: "GreenCorp announces its new line of eco-friendly cleaning products launching this quarter."
</CONTEXT>
<CLAIM>GreenCorp's new cleaning products are made from 100% biodegradable ingredients.</CLAIM>
<OUTPUT>
Label: abstain
Type: N/A
Justification: The provided context mentions "eco-friendly cleaning products" but contains no specific information about ingredients, biodegradability testing, or product composition. Without this information, I cannot verify whether the "100% biodegradable ingredients" claim is factually accurate or false.
</OUTPUT>
</EXAMPLE 6>
 
### EVALUATION INPUT
 
<CONTEXT>
[Context may be provided here, or this section may be empty]
</CONTEXT>
 
<CLAIM>
Claim: The claim at hand
</CLAIM>
 
<OUTPUT>
Label:
Type:
Justification:
</OUTPUT>

company: {company}
claim  : "{claim}"

Context from knowledge graph:
{context}
"""

FEW_TMPL_OLD = """
You are an **ESG (Environmental, Social, and Governance) and Greenwashing Fact-Checker.**

1. **Type 1 - Misleading labels:** Using vague, unverifiable, or misleading sustainability labels.  
2. **Type 2 - Legal obligations as achievements:** Presenting compliance with laws or regulations as an environmental achievement.  
3. **Type 3 - Overgeneralization:** Claiming a product, service, or company is sustainable when only part of it meets sustainability criteria.  
4. **Type 4 - Unsupported claims:** Making environmental claims that cannot be substantiated with evidence.

**Task:** Given the following text, fact-check the sustainability-related statement and determine whether each constitutes **greenwashing**, **not_greenwashing**, or if you should **abstain** (when evidence is insufficient).
Return valid **JSON only** with this shape:
{{
  "label": "greenwashing" | "not_greenwashing" | "abstain",
  "reasoning": "<reasoning for your decision>",
  "cited_node_ids": [int,...]   // may be empty
}}

<EXAMPLE 1>
<CLAIM>
H&M promotes its new "Conscious Collection" as a sustainable fashion line.
</CLAIM>
<OUTPUT>
{{
    "label": "greenwashing",
    "reasoning": "The 'Conscious' label used by H&M is vague and lacks clearly defined or verifiable sustainability criteria, making the claim potentially misleading to consumers.",
    "cited_node_ids": []
}},
</OUTPUT>
</EXAMPLE 1>
--- 
<EXAMPLE 2>
<CLAIM>
 McDonald's announces it is reducing plastic waste by introducing reusable cutlery in France.
</CLAIM>
 <OUTPUT>
{{
    "label": "greenwashing",
    "reasoning": "McDonald's presented the introduction of reusable cutlery in France as a voluntary sustainability effort, whereas it was mandated by French legislation in 2023, making the framing misleading.",
    "cited_node_ids": [ ... ]
}},
</OUTPUT>
</EXAMPLE 2>
 ---
 <EXAMPLE 3>
 <CLAIM>
Walmart advertises its "eco-friendly" bamboo towels as fully sustainable.
</CLAIM>
 <OUTPUT>
{{
    "label": "greenwashing",
    "reasoning": "Walmart's claim that its bamboo towels are fully sustainable is misleading because they are produced from rayon derived from bamboo, a chemical-intensive process that negates most of the material’s natural environmental advantages.",
    "cited_node_ids": []
}},
</OUTPUT>
</EXAMPLE 3>
 ---
<EXAMPLE 4>
<CLAIM>
Keurig states that all of its single-use coffee pods are "easily recyclable in any city recycling program."
</CLAIM>
 <OUTPUT>
{{
    "label": "greenwashing",
    "reasoning": "Keurig’s claim that its single-use coffee pods are easily recyclable in any city program is false; the company was fined for this misleading statement since most municipal recycling facilities cannot process its pods.",
    "cited_node_ids": []
}},
</OUTPUT> 
</EXAMPLE 4>
 ---
<EXAMPLE 5>
<CLAIM>
Tesla reports that its Gigafactory in Nevada now runs entirely on renewable energy.
</CLAIM>
<OUTPUT>
{{
    "label": "not_greenwashing",
    "reasoning": "Tesla’s report that its Gigafactory in Nevada runs entirely on renewable energy is substantiated by verifiable data and represents a genuine sustainability milestone rather than a misleading marketing claim.",
    "cited_node_ids": []
}}
</OUTPUT>
</EXAMPLE 5>

company: {company}
claim  : "{claim}"

Context from knowledge graph:
{context}
"""

ZERO_TMPL = """
You are an **ESG (Environmental, Social, and Governance) and Greenwashing Fact-Checker.**
 
**Task:** Given an ESG-related claim, fact-check it and determine whether it constitutes **greenwashing**, **not_greenwashing**, or if you should **abstain**.
 
### Fact-Checking Process
 
**When context is provided below:** Use the provided context as your PRIMARY source of truth. Cross-reference claims against specific information in the context.
 
**When no context is provided:** Use your internal knowledge to verify factual accuracy. You may only make determinations based on facts you can confidently recall.
 
**In BOTH cases:** Your decision MUST be based on verifiable facts that either prove or disprove the claim. NEVER judge based on:
- Vague wording or marketing language
- How "suspicious" or "too good to be true" something sounds  
- Absence of detail (this warrants abstain, not greenwashing)
- Your intuition or reasoning about what "could be" misleading
 
### Decision Criteria
 
**Greenwashing** - You have specific evidence that the claim is factually false or misleading:
1. **Type 1 - Factually false labels:** You can verify the label/certification is false, doesn't exist, or misrepresents actual standards.
2. **Type 2 - Legal obligations misrepresented as voluntary:** You can verify the action was legally mandated, not a voluntary initiative.
3. **Type 3 - Partial truth presented as whole:** You can verify only part of the product/service meets the stated criteria.
4. **Type 4 - Claims contradicted by evidence:** You have specific facts/evidence that directly contradict the claim.
 
**Not Greenwashing** - You can verify the claim is factually accurate based on available information (context or knowledge).
 
**Abstain** - You cannot verify factual accuracy either way. Use this when:
- No context is provided AND you lack sufficient internal knowledge
- Context is provided but doesn't contain information to verify/refute the specific claim
- You cannot confidently determine if the claim is true or false
 
### Output Format (strictly follow this, no extra text)
 
Label: <greenwashing / not_greenwashing / abstain>
Type: <Type 1 / Type 2 / Type 3 / Type 4 / N/A>
Justification: <If using context: Quote specific passages and explain how they prove/disprove the claim. If using internal knowledge: State specific facts you know and their sources when possible. If abstaining: State what information is missing.>
 
### EVALUATION INPUT
 
<CONTEXT>
[Context may be provided here, or this section may be empty]
</CONTEXT>
 
<CLAIM>
Claim: The claim at hand
</CLAIM>
 
<OUTPUT>
Label:
Type:
Justification:
</OUTPUT>

company: {company}
claim  : "{claim}"

Context from knowledge graph:
{context}
"""

ZERO_TMPL_OLD = """
You are an **ESG (Environmental, Social, and Governance) and Greenwashing Fact-Checker.**

1. **Type 1 - Misleading labels:** Using vague, unverifiable, or misleading sustainability labels.  
2. **Type 2 - Legal obligations as achievements:** Presenting compliance with laws or regulations as an environmental achievement.  
3. **Type 3 - Overgeneralization:** Claiming a product, service, or company is sustainable when only part of it meets sustainability criteria.  
4. **Type 4 - Unsupported claims:** Making environmental claims that cannot be substantiated with evidence.

**Task:** Given the following text, fact-check the sustainability-related statement and determine whether each constitutes **greenwashing**, **not_greenwashing**, or if you should **abstain** (when evidence is insufficient).
Return valid **JSON only** with this shape:
{{
  "label": "greenwashing" | "not_greenwashing" | "abstain",
  "reasoning": "<reasoning for your decision>",
  "cited_node_ids": [int,...]   // may be empty
}}

company: {company}
claim  : "{claim}"

Context from knowledge graph:
{context}
"""


prompt_from_fig7_refined = """
 
You are an **ESG (Environmental, Social, and Governance) and Greenwashing Fact-Checker.**
 
**Task:** Given an ESG-related claim, fact-check it and determine whether it constitutes **greenwashing**, **not_greenwashing**, or if you should **abstain**.
 
### Fact-Checking Process
 
**When context is provided below:** Use the provided context as your PRIMARY source of truth. Cross-reference claims against specific information in the context.
 
**When no context is provided:** Use your internal knowledge to verify factual accuracy. You may only make determinations based on facts you can confidently recall.
 
**In BOTH cases:** Your decision MUST be based on verifiable facts that either prove or disprove the claim. NEVER judge based on:
- Vague wording or marketing language
- How "suspicious" or "too good to be true" something sounds  
- Absence of detail (this warrants abstain, not greenwashing)
- Your intuition or reasoning about what "could be" misleading
 
### Decision Criteria
 
**Greenwashing** - You have specific evidence that the claim is factually false or misleading:
1. **Type 1 - Factually false labels:** You can verify the label/certification is false, doesn't exist, or misrepresents actual standards.
2. **Type 2 - Legal obligations misrepresented as voluntary:** You can verify the action was legally mandated, not a voluntary initiative.
3. **Type 3 - Partial truth presented as whole:** You can verify only part of the product/service meets the stated criteria.
4. **Type 4 - Claims contradicted by evidence:** You have specific facts/evidence that directly contradict the claim.
 
**Not Greenwashing** - You can verify the claim is factually accurate based on available information (context or knowledge).
 
**Abstain** - You cannot verify factual accuracy either way. Use this when:
- No context is provided AND you lack sufficient internal knowledge
- Context is provided but doesn't contain information to verify/refute the specific claim
- You cannot confidently determine if the claim is true or false
 
### Output Format (strictly follow this, no extra text)
 
Label: <greenwashing / not_greenwashing / abstain>
Type: <Type 1 / Type 2 / Type 3 / Type 4 / N/A>
Justification: <If using context: Quote specific passages and explain how they prove/disprove the claim. If using internal knowledge: State specific facts you know and their sources when possible. If abstaining: State what information is missing.>
 
### EVALUATION INPUT
 
<CONTEXT>
[Context may be provided here, or this section may be empty]
</CONTEXT>
 
<CLAIM>
Claim: The claim at hand
</CLAIM>
 
<OUTPUT>
Label:
Type:
Justification:
</OUTPUT>
 
"""
 
prompt_from_fig8_refined = """
 
You are an **ESG (Environmental, Social, and Governance) and Greenwashing Fact-Checker.**
 
**Task:** Given an ESG-related claim, fact-check it and determine whether it constitutes **greenwashing**, **not_greenwashing**, or if you should **abstain**.
 
### Fact-Checking Process
 
**When context is provided below:** Use the provided context as your PRIMARY source of truth. Cross-reference claims against specific information in the context.
 
**When no context is provided:** Use your internal knowledge to verify factual accuracy. You may only make determinations based on facts you can confidently recall.
 
**In BOTH cases:** Your decision MUST be based on verifiable facts that either prove or disprove the claim. NEVER judge based on:
- Vague wording or marketing language
- How "suspicious" or "too good to be true" something sounds  
- Absence of detail (this warrants abstain, not greenwashing)
- Your intuition or reasoning about what "could be" misleading
 
### Decision Criteria
 
**Greenwashing** - You have specific evidence that the claim is factually false or misleading:
1. **Type 1 - Factually false labels:** You can verify the label/certification is false, doesn't exist, or misrepresents actual standards.
2. **Type 2 - Legal obligations misrepresented as voluntary:** You can verify the action was legally mandated, not a voluntary initiative.
3. **Type 3 - Partial truth presented as whole:** You can verify only part of the product/service meets the stated criteria.
4. **Type 4 - Claims contradicted by evidence:** You have specific facts/evidence that directly contradict the claim.
 
**Not Greenwashing** - You can verify the claim is factually accurate based on available information (context or knowledge).
 
**Abstain** - You cannot verify factual accuracy either way. Use this when:
- No context is provided AND you lack sufficient internal knowledge
- Context is provided but doesn't contain information to verify/refute the specific claim
- You cannot confidently determine if the claim is true or false
 
### FEW-SHOT EXAMPLES
 
<EXAMPLE 1: No context provided, insufficient internal knowledge>
<CONTEXT>
[No context provided]
</CONTEXT>
<CLAIM>H&M promotes its new "Conscious Collection" as a sustainable fashion line.</CLAIM>
<OUTPUT>
Label: abstain
Type: N/A
Justification: No context provided. While I know H&M has a "Conscious Collection," I do not have specific, verifiable information about the materials, certifications, or manufacturing processes used in this collection to determine if the sustainability claims are factually accurate or false.
</OUTPUT>
</EXAMPLE 1>
---
<EXAMPLE 2: No context provided, using internal knowledge>
<CONTEXT>
[No context provided]
</CONTEXT>
<CLAIM>McDonald's announces it is reducing plastic waste by introducing reusable cutlery in France.</CLAIM>
<OUTPUT>
Label: greenwashing
Type: Type 2
Justification: Based on internal knowledge: The French Anti-Waste Law (AGEC), passed in February 2020, mandates that fast-food restaurants must use reusable tableware for on-site dining as of January 1, 2023. McDonald's is legally required to provide reusable cutlery in France, making this a legal compliance issue rather than a voluntary sustainability initiative.
</OUTPUT>
</EXAMPLE 2>
---
<EXAMPLE 3: Context provided, contradicts claim>
<CONTEXT>
Product manufacturing report: "Walmart's bamboo towel line uses bamboo viscose, produced through a chemical process involving carbon disulfide (CS2) and sodium hydroxide (NaOH). The manufacturing process generates significant chemical waste and requires substantial water usage."
</CONTEXT>
<CLAIM>Walmart advertises its "eco-friendly" bamboo towels as fully sustainable.</CLAIM>
<OUTPUT>
Label: greenwashing
Type: Type 3
Justification: The provided context shows that while bamboo is the raw material, the manufacturing process involves "carbon disulfide (CS2) and sodium hydroxide (NaOH)" and "generates significant chemical waste and requires substantial water usage." The claim of "fully sustainable" is factually inaccurate as it ignores the environmentally harmful manufacturing process documented in the context.
</OUTPUT>
</EXAMPLE 3>
---
<EXAMPLE 4: Context provided, supports claim with evidence>
<CONTEXT>
Regulatory filing: "Canada's Competition Bureau issued a penalty to Keurig Canada in January 2022 for making false or misleading claims about the recyclability of its K-Cup pods. Investigation found that the majority of Canadian municipalities do not accept K-Cup pods in curbside recycling programs due to their multi-material composition."
</CONTEXT>
<CLAIM>Keurig states that all of its single-use coffee pods are "easily recyclable in any city recycling program."</CLAIM>
<OUTPUT>
Label: greenwashing
Type: Type 4
Justification: The context directly contradicts the claim, stating that "the majority of Canadian municipalities do not accept K-Cup pods in curbside recycling programs" and that Keurig was penalized by Canada's Competition Bureau in 2022 "for making false or misleading claims about the recyclability" of these pods.
</OUTPUT>
</EXAMPLE 4>
---
<EXAMPLE 5: Context provided, verifies claim>
<CONTEXT>
Energy audit report: "Tesla's Gigafactory 1 in Nevada operates with a 70 MW solar array installed on its roof, supplemented by renewable energy purchased through power purchase agreements (PPAs). As of Q4 2023, 100% of the facility's energy consumption comes from renewable sources, verified by third-party auditor SGS."
</CONTEXT>
<CLAIM>Tesla reports that its Gigafactory in Nevada now runs entirely on renewable energy.</CLAIM>
<OUTPUT>
Label: not_greenwashing
Type: N/A
Justification: The context verifies the claim with specific details: "70 MW solar array" on-site, renewable energy PPAs, and third-party verification by SGS confirming "100% of the facility's energy consumption comes from renewable sources." The claim is factually accurate based on documented evidence.
</OUTPUT>
</EXAMPLE 5>
---
<EXAMPLE 6: Context provided but insufficient>
<CONTEXT>
Press release: "GreenCorp announces its new line of eco-friendly cleaning products launching this quarter."
</CONTEXT>
<CLAIM>GreenCorp's new cleaning products are made from 100% biodegradable ingredients.</CLAIM>
<OUTPUT>
Label: abstain
Type: N/A
Justification: The provided context mentions "eco-friendly cleaning products" but contains no specific information about ingredients, biodegradability testing, or product composition. Without this information, I cannot verify whether the "100% biodegradable ingredients" claim is factually accurate or false.
</OUTPUT>
</EXAMPLE 6>
 
### EVALUATION INPUT
 
<CONTEXT>
[Context may be provided here, or this section may be empty]
</CONTEXT>
 
<CLAIM>
Claim: The claim at hand
</CLAIM>
 
<OUTPUT>
Label:
Type:
Justification:
</OUTPUT>
"""

_RX_PUNCT = re.compile(r"[\\W_]+", re.U)

CFG_JSON = types.GenerateContentConfig(
    response_mime_type="application/json",
    temperature=0.0,
)


def _clean_name(n: Any) -> str:
    """
    Normalise company names for fuzzy matching.
    Returns '' for None / non‑str inputs.
    """
    if not isinstance(n, str) or not n.strip():
        return ""

    n = unicodedata.normalize("NFKD", n).encode("ascii", "ignore").decode()
    n = _RX_PUNCT.sub(" ", n.lower()).strip()
    tokens = [t for t in n.split() if t not in _CORP_SUFFIXES]
    return " ".join(tokens)


FUZZY_MIN = 65
EMBED_MIN = 0.7


def get_company_id(
    name: str, *, fuzzy_min: int = FUZZY_MIN, embed_min: float = EMBED_MIN
) -> int | None:
    if not name:
        return None

    target_clean = _clean_name(name)
    print(f"DEBUG: Input '{name}' cleaned to '{repr(target_clean)}'")

    with driver.session() as sess:
        rows = sess.run(
            "MATCH (o:Organization) RETURN id(o) AS id, o.name AS name"
        ).data()

    orgs = []
    for row in rows:
        if row["name"]:
            cleaned_db_name = _clean_name(row["name"])
            orgs.append((row["id"], row["name"], cleaned_db_name))
            print(
                f"DEBUG: DB name '{row['name']}' cleaned to '{repr(cleaned_db_name)}'"
            )

    for oid, raw, clean in orgs:
        if clean == target_clean:
            print(f"MATCH FOUND (Exact Clean): '{name}' -> '{raw}' (ID: {oid})")
            return oid

    containment_candidates = []
    for oid, raw, clean in orgs:
        if target_clean and clean and target_clean in clean:
            containment_candidates.append((oid, raw, clean))

    if len(containment_candidates) == 1:
        oid, raw, clean = containment_candidates[0]
        print(f"MATCH FOUND (Unique Containment): '{name}' -> '{raw}' (ID: {oid})")
        return oid
    elif len(containment_candidates) > 1:
        best_candidate = None
        best_contain_score = -1

        print(f"DEBUG: Multiple containment candidates for '{name}':")
        for oid, raw, clean in containment_candidates:
            score = fuzz.token_sort_ratio(target_clean, clean)
            print(f"  - Candidate: '{raw}' (Cleaned: '{repr(clean)}') - Score: {score}")

            if score > best_contain_score:
                best_contain_score = score
                best_candidate = (oid, raw, clean)

        if best_candidate and best_contain_score >= fuzzy_min:
            oid, raw, clean = best_candidate
            print(
                f"MATCH FOUND (Best Containment): '{name}' -> '{raw}' (ID: {oid}) with score {best_contain_score}"
            )
            return oid
        else:
            print(f"DEBUG: No strong containment candidate found, falling back.")

    best_id_fuzzy, best_score_fuzzy = None, 0
    for oid, raw, clean in orgs:
        score = fuzz.token_set_ratio(target_clean, clean)
        if score > best_score_fuzzy:
            best_id_fuzzy, best_score_fuzzy = oid, score

    print(
        f"DEBUG: Best fuzzy match: ID={best_id_fuzzy}, Score={best_score_fuzzy} (threshold={fuzzy_min})"
    )

    if best_score_fuzzy >= fuzzy_min:
        matched_name = next(raw for oid, raw, clean in orgs if oid == best_id_fuzzy)
        print(f"MATCH FOUND (Fuzzy): '{name}' -> '{matched_name}' (ID: {best_id_fuzzy})")
        return best_id_fuzzy

    target_emb = encoder.encode(name, normalize_embeddings=True)
    best_id_emb, best_sim = None, 0.0

    for oid, raw, _ in orgs:
        db_emb = encoder.encode(raw, normalize_embeddings=True)
        sim = dot(target_emb, db_emb) / (norm(target_emb) * norm(db_emb))
        if sim > best_sim:
            best_id_emb, best_sim = oid, sim

    print(
        f"DEBUG: Best embedding match: ID={best_id_emb}, Sim={best_sim:.3f} (threshold={embed_min})"
    )

    if best_sim >= embed_min:
        matched_name = next(raw for oid, raw, _ in orgs if oid == best_id_emb)
        print(f"MATCH FOUND (Embedding): '{name}' -> '{matched_name}' (ID: {best_id_emb})")
        return best_id_emb

    print(f"NO MATCH for '{name}'")
    return None


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors"""
    return dot(vec1, vec2) / (norm(vec1) * norm(vec2))


def load_claim_nodes(claim_idx: int, claims_dir: str = "big_dataset") -> Optional[List[Dict[str, Any]]]:
    """
    Load nodes with embeddings from claim_{i}.json file.
    
    Args:
        claim_idx: The claim index
        claims_dir: Directory containing claim_{i}.json files
    
    Returns:
        List of nodes with embeddings, or None if file not found
    """
    claim_file = Path(claims_dir) / f"claim_{claim_idx}.json"
    
    if not claim_file.exists():
        print(f"Warning: Claim file not found: {claim_file}")
        return None
    
    try:
        with open(claim_file, 'r') as f:
            data = json.load(f)
        
        nodes = data.get('nodes', [])
        
        # Verify embeddings exist
        nodes_with_embeddings = [n for n in nodes if 'embedding' in n]
        
        if len(nodes_with_embeddings) < len(nodes):
            print(f"Warning: Only {len(nodes_with_embeddings)}/{len(nodes)} nodes have embeddings in {claim_file.name}")
        
        return nodes_with_embeddings
    except Exception as e:
        print(f"Error loading {claim_file}: {e}")
        return None
    
def _safe_json(text: str) -> Dict[str, Any]:
    def _clean_text(s: str) -> str:
        """Remove markdown and extra whitespace."""
        s = s.strip()
        # Remove markdown code blocks
        if s.startswith("```"):
            s = re.sub(r'^```[a-z]*\n?', '', s)
            s = re.sub(r'```$', '', s)
        # Remove <OUTPUT> tags if present
        s = re.sub(r'<OUTPUT>\s*', '', s, flags=re.IGNORECASE)
        s = re.sub(r'\s*</OUTPUT>', '', s, flags=re.IGNORECASE)
        return s.strip()
    
    def _extract_label(s: str) -> str | None:
        """Extract label value."""
        patterns = [
            # Standard format: Label: greenwashing
            r'Label:\s*\*{0,2}(greenwashing|not_greenwashing|abstain)\*{0,2}',
            # With markdown: Label: **greenwashing**
            r'Label:\s*\*\*(greenwashing|not_greenwashing|abstain)\*\*',
            # Standalone with asterisks
            r'\*\*(greenwashing|not_greenwashing|abstain)\*\*',
            # Just the word after "Label:"
            r'Label:\s*(greenwashing|not_greenwashing|abstain)',
            # Quoted
            r'Label:\s*["\']?(greenwashing|not_greenwashing|abstain)["\']?',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, s, re.IGNORECASE | re.MULTILINE)
            if match:
                label = match.group(1).lower().strip()
                # Normalize variants
                if label in ['greenwashing', 'not_greenwashing', 'abstain']:
                    return label
        
        return None
    
    def _extract_type(s: str) -> str | None:
        """Extract type value."""
        patterns = [
            # Standard format: Type: Type 1
            r'Type:\s*\*{0,2}(Type\s*[1-4]|N/A)\*{0,2}',
            # With markdown
            r'Type:\s*\*\*(Type\s*[1-4]|N/A)\*\*',
            # Just the type
            r'Type:\s*(Type\s*[1-4]|N/A)',
            # Quoted
            r'Type:\s*["\']?(Type\s*[1-4]|N/A)["\']?',
            # Without "Type" prefix: Type: 1
            r'Type:\s*\*{0,2}([1-4])\*{0,2}(?!\d)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, s, re.IGNORECASE | re.MULTILINE)
            if match:
                type_val = match.group(1).strip()
                # Normalize
                if type_val.upper() == 'N/A':
                    return 'N/A'
                # If just a number, add "Type" prefix
                if type_val.isdigit():
                    return f'Type {type_val}'
                # Normalize spacing: "Type1" -> "Type 1"
                type_val = re.sub(r'Type\s*(\d)', r'Type \1', type_val, flags=re.IGNORECASE)
                return type_val
        
        return None
    
    def _extract_justification(s: str) -> str:
        """Extract justification text - handles multi-line responses."""
        
        # Find the start of justification
        match = re.search(r'Justification:\s*', s, re.IGNORECASE | re.MULTILINE)
        if not match:
            return ""
        
        start_pos = match.end()
        
        # Extract everything after "Justification:"
        remaining_text = s[start_pos:]
        
        # Find where it ends (next field label or end of string)
        end_markers = [
            r'\n\s*Label:',
            r'\n\s*Type:',
            r'</OUTPUT>',
            r'</',  # Any closing tag
        ]
        
        end_pos = len(remaining_text)
        for marker in end_markers:
            marker_match = re.search(marker, remaining_text, re.IGNORECASE)
            if marker_match:
                end_pos = min(end_pos, marker_match.start())
        
        justification = remaining_text[:end_pos].strip()
        
        # Clean up
        justification = justification.strip('"\'')
        justification = re.sub(r'[ \t]+', ' ', justification)  # Normalize spaces on same line
        justification = justification.strip()
        
        return justification if len(justification) > 3 else ""
    
    def _try_structured_parse(s: str) -> Dict[str, Any] | None:
        """Try to extract all three fields in order."""
        result = {}
        
        # Extract each field
        label = _extract_label(s)
        type_val = _extract_type(s)
        justification = _extract_justification(s)
        
        # Must have at least label to be valid
        if label:
            result['label'] = label
            result['type'] = type_val if type_val else 'N/A'
            result['reasoning'] = justification if justification else ''
            return result
        
        return None
    
    def _try_json_fallback(s: str) -> Dict[str, Any] | None:
        """
        Fallback: try to parse as JSON (in case LLM ignored format instructions).
        """
        try:
            # Remove markdown
            s = re.sub(r'^```json\s*', '', s)
            s = re.sub(r'^```\s*', '', s)
            s = re.sub(r'```$', '', s)
            
            # Try direct JSON parse
            data = json.loads(s.strip())
            
            if isinstance(data, dict):
                # Map JSON keys to expected keys
                result = {}
                
                # Label
                for key in ['label', 'Label', 'LABEL']:
                    if key in data:
                        result['label'] = str(data[key]).lower()
                        break
                
                # Type
                for key in ['type', 'Type', 'TYPE']:
                    if key in data:
                        result['type'] = str(data[key])
                        break
                
                # Justification
                for key in ['justification', 'Justification', 'JUSTIFICATION', 'reasoning']:
                    if key in data:
                        result['reasoning'] = str(data[key])
                        break
                
                if 'label' in result:
                    result.setdefault('type', 'N/A')
                    result.setdefault('reasoning', '')
                    return result
        
        except (json.JSONDecodeError, ValueError):
            pass
        
        return None
    
    # Validate input
    if not text or not isinstance(text, str):
        return {
            "label": "error_parsing",
            "type": "N/A",
            "reasoning": "Empty or invalid response text"
        }
    
    original_text = text
    text = _clean_text(text)
    
    # Strategy 1: Structured text parsing (primary)
    result = _try_structured_parse(text)
    if result:
        return result
    
    # Strategy 2: Try with original text
    result = _try_structured_parse(original_text)
    if result:
        return result
    
    # Strategy 3: JSON fallback (in case LLM went rogue)
    result = _try_json_fallback(text)
    if result:
        return result
    
    result = _try_json_fallback(original_text)
    if result:
        return result
    
    # All strategies failed
    return {
        "label": "error_parsing",
        "type": "N/A",
        "reasoning": f"Could not parse LLM response. Raw: {original_text[:300]}"
    }

def format_entities_as_json(entities: List[Dict[str, Any]], company_name: str) -> str:
    """
    Format entities with their paths as a structured JSON optimized for LLM consumption.
    Handles both path-based evidence (from retrieve_evidence_with_paths) and 
    direct connection evidence (from retrieve_evidence).
    """
    def convert_to_serializable(obj):
        """Convert numpy types to native Python types for JSON serialization."""
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        return obj
    
    def get_node_display_name(node_type: str, properties: Dict[str, Any]) -> str:
        """Get a display name using class and first meaningful attribute."""
        # Get first non-internal property value
        for k, v in sorted(properties.items()):
            if k not in ['embedding', 'id', 'valid_from', 'valid_to', 'is_current'] and v is not None:
                # Truncate long values
                str_v = str(v)
                return f"{node_type}({str_v})"
        return node_type
    
    def format_path(path_data: Dict[str, Any]) -> Dict[str, Any]:
        """Format path with complete node information."""
        path_nodes = path_data.get('path_nodes', [])
        rel_types = path_data.get('rel_types', [])
        path_length = path_data.get('path_length', 0)
        
        # Format each node with full properties
        formatted_nodes = []
        for node_data in path_nodes:
            node_props = node_data.get('properties', {})
            node_labels = node_data.get('labels', [])
            
            # Filter out internal properties
            filtered_props = {}
            for k, v in sorted(node_props.items()):
                if k not in ['embedding', 'id', 'valid_from', 'valid_to', 'is_current'] and v is not None:
                    filtered_props[k] = convert_to_serializable(v)
            
            formatted_nodes.append({
                "type": node_labels[0] if node_labels else "Node",
                "properties": filtered_props
            })
        
        # Create a readable summary using class + first attribute
        path_summary_parts = []
        for i, node in enumerate(formatted_nodes):
            node_type = node["type"]
            props = node["properties"]
            display_name = get_node_display_name(node_type, props)
            path_summary_parts.append(display_name)
            
            if i < len(rel_types):
                path_summary_parts.append(f"-[{rel_types[i]}]->")
        
        return {
            "summary": " ".join(path_summary_parts),
            "hops": path_length,
            "relationships": rel_types,
            "nodes": formatted_nodes
        }
    
    def format_direct_connection(entity_label: str, entity_props: Dict[str, Any], 
                                 rel_type: str, company_name: str) -> Dict[str, Any]:
        """Format a direct connection (1-hop) when no path data is available."""
        # Filter entity properties
        filtered_entity_props = {}
        for k, v in sorted(entity_props.items()):
            if k not in ['embedding', 'id', 'valid_from', 'valid_to', 'is_current'] and v is not None:
                filtered_entity_props[k] = convert_to_serializable(v)
        
        # Create summary
        entity_display = get_node_display_name(entity_label, filtered_entity_props)
        summary = f"Organization({company_name}) -[{rel_type}]-> {entity_display}"
        
        return {
            "summary": summary,
            "hops": 1,
            "relationships": [rel_type],
            "nodes": [
                {
                    "type": "Organization",
                    "properties": {"name": company_name}
                },
                {
                    "type": entity_label,
                    "properties": filtered_entity_props
                }
            ]
        }
    
    # Group entities by type for better organization
    entities_by_type = {}
    for entity in entities:
        labels = entity.get("labels", [])
        label = labels[0] if labels else "Unknown"
        
        if label not in entities_by_type:
            entities_by_type[label] = []
        
        props = entity.get("properties", {})
        similarity = entity.get("similarity", 0.0)
        paths = entity.get("paths", [])  # May be empty or missing
        rel_type = entity.get("rel_type", None)  # For direct connections
        
        # Filter properties
        filtered_props = {}
        for k, v in sorted(props.items()):
            if k not in ['embedding', 'id', 'valid_from', 'valid_to', 'is_current'] and v is not None:
                filtered_props[k] = convert_to_serializable(v)
        
        # Determine connection format based on available data
        if paths:
            # Has path data (from retrieve_evidence_with_paths)
            shortest_path = min(paths, key=lambda p: p.get('path_length', 999), default=None)
            connection = format_path(shortest_path) if shortest_path else None
        elif rel_type is not None:
            # Has direct connection (from retrieve_evidence)
            connection = format_direct_connection(label, filtered_props, rel_type, company_name)
        else:
            connection = None
        
        entity_obj = {
            "properties": filtered_props,
            "connection": connection,
        }
        
        entities_by_type[label].append(entity_obj)
    
    # Sort entities within each type by properties (for deterministic ordering)
    for label in entities_by_type:
        entities_by_type[label].sort(key=lambda x: str(x['properties']))
    
    # Build final structure with clear hierarchy
    formatted_data = {
        "company": company_name,
        "evidence_summary": {
            "total_entities": len(entities),
            "entity_types": {label: len(ents) for label, ents in entities_by_type.items()}
        },
        "evidence_by_type": entities_by_type
    }
    
    return json.dumps(formatted_data, indent=2, ensure_ascii=False)

def retrieve_evidence_with_paths(
    company_id: int,
    claim: str,
    claim_idx: int = 0,
    top_k_per_class: int = 10,
    similarity_threshold: float = 0.3,
    max_hops: int = 3,
    embeddings_dir: str = "big_dataset"
) -> List[Dict[str, Any]]:
    """
    Retrieve evidence by:
    1. Load nodes with embeddings from claim_{i}.json
    2. For each JSON node, find unique nodes of same class within k-hop neighborhood
    3. Calculate similarities on unique nodes only
    4. Keep top-k most similar nodes and fetch FULL PATHS to them
    5. For ALWAYS_INCLUDE_CLASSES not in JSON, search k-hop neighborhood
    6. Return paths with all nodes and relationships
    """
    print(f"\n=== Retrieving evidence (k={max_hops} hops) for company_id={company_id}, claim_idx={claim_idx} ===")
    
    json_nodes = load_claim_nodes(claim_idx, embeddings_dir)
    
    if not json_nodes:
        print(f"No nodes found in claim_{claim_idx}.json")
        json_nodes = []
    else:
        print(f"Loaded {len(json_nodes)} nodes with embeddings from claim_{claim_idx}.json")
        json_nodes = sorted(json_nodes, key=lambda n: (
            n.get('class', 'Unknown'),
            str(n.get('properties', {}).get('name', '')),
            str(n.get('properties', {}).get('description', '')),
            str(n.get('properties', {}))
        ))
    
    json_classes = set()
    evidence_paths = []
    
    with driver.session() as session:
        # Process each JSON node - find similar nodes in k-hop neighborhood
        for idx, json_node in enumerate(json_nodes):
            node_class = json_node.get('class', 'Unknown')
            # if node_class is KPIObservation or Penalty, skip here (handled later)
            if node_class in ALWAYS_INCLUDE_CLASSES:
                continue
            json_classes.add(node_class)
            json_embedding = np.array(json_node['embedding'], dtype=np.float32)
            
            print(f"\n[{idx+1}/{len(json_nodes)}] Processing JSON node of class: {node_class}")
            
            # STAGE 1: Get all UNIQUE nodes of this class within k hops
            query_nodes = f"""
            MATCH (company:Organization)-[*1..{max_hops}]-(node:{node_class})
            WHERE id(company) = $company_id
            RETURN DISTINCT
                id(node) as node_id,
                node,
                labels(node) as labels
            ORDER BY node_id
            """
            
            try:
                result = session.run(query_nodes, company_id=company_id).data()
                
                if not result:
                    print(f"  No {node_class} nodes found within {max_hops} hops")
                    continue
                
                print(f"  Found {len(result)} unique {node_class} nodes within {max_hops} hops, calculating similarities...")
                
                # Calculate similarity for each unique node
                node_similarities = []
                for record in result:
                    node = record['node']
                    node_id = record['node_id']
                    
                    # Create text representation
                    node_dict = dict(node)
                    sorted_props = sorted(node_dict.items())
                    
                    node_text_parts = [f"{node_class}:"]
                    for key, value in sorted_props:
                        if value is not None and key not in ['valid_from', 'valid_to', 'is_current', 'embedding']:
                            node_text_parts.append(f"{key}: {value}")
                    node_text = " | ".join(node_text_parts)
                    
                    graph_node_embedding = encoder.encode(node_text, normalize_embeddings=True)
                    similarity = cosine_similarity(json_embedding, graph_node_embedding)
                    
                    if similarity >= similarity_threshold:
                        node_similarities.append({
                            'node_id': node_id,
                            'node': node,
                            'labels': record['labels'],
                            'similarity': similarity
                        })
                
                # Sort and take top-k BEFORE fetching paths
                node_similarities.sort(key=lambda x: (-x['similarity'], x['node_id']))
                top_nodes = node_similarities[:top_k_per_class]
                
                print(f"  Selected {len(top_nodes)} most similar nodes (out of {len(node_similarities)} above threshold)")
                
                # STAGE 2: Fetch paths ONLY for top-k nodes
                for item in top_nodes:
                    query_paths = f"""
                    MATCH (company:Organization)
                    WHERE id(company) = $company_id
                    MATCH (node:{node_class})
                    WHERE id(node) = $node_id
                    MATCH path = shortestPath((company)-[*1..{max_hops}]-(node))
                    WITH path,
                         [rel in relationships(path) | type(rel)] as rel_types,
                         [n in nodes(path) | {{id: id(n), labels: labels(n), properties: properties(n)}}] as path_nodes
                    RETURN 
                        path_nodes,
                        rel_types,
                        length(path) as path_length
                    LIMIT 1
                    """
                    
                    path_result = session.run(query_paths, company_id=company_id, node_id=item['node_id']).data()
                    
                    if path_result:
                        evidence_dict = {
                            'node_id': item['node_id'],
                            'labels': sorted(item['labels']),
                            'properties': dict(item['node']),
                            'similarity': item['similarity'],
                            'embedding_source': f'similar to JSON node #{idx+1}',
                            'paths': [{
                                'path_nodes': path_result[0]['path_nodes'],
                                'rel_types': path_result[0]['rel_types'],
                                'path_length': path_result[0]['path_length']
                            }]
                        }
                        evidence_paths.append(evidence_dict)
                        
                        node_name = dict(item['node']).get('name', 
                                       dict(item['node']).get('description', 
                                       dict(item['node']).get('target', 'unnamed')))
                        print(f"    {node_class}: {str(node_name)[:50]}... "
                              f"(similarity={item['similarity']:.3f}, "
                              f"path_length={path_result[0]['path_length']} hops)")
                
            except Exception as e:
                print(f"  Error querying Neo4j for {node_class}: {e}")
                continue
        
        # Handle ALWAYS_INCLUDE_CLASSES not in JSON
        missing_always_include = sorted([cls for cls in ALWAYS_INCLUDE_CLASSES 
                                        if cls not in json_classes])
        
        if missing_always_include:
            print(f"\n=== Processing ALWAYS_INCLUDE_CLASSES not in JSON: {missing_always_include} ===")
            claim_embedding = encoder.encode(claim, normalize_embeddings=True)
            temp = top_k_per_class
            for class_name in missing_always_include:
                if class_name == "KPIObservation":
                    top_k_per_class=3
                else:
                    top_k_per_class=temp
                print(f"\nProcessing ALWAYS_INCLUDE class: {class_name}")
                
                # STAGE 1: Get unique nodes
                query_nodes = f"""
                MATCH (company:Organization)-[*1..1]-(node:{class_name})
                WHERE id(company) = $company_id
                RETURN DISTINCT
                    id(node) as node_id,
                    node,
                    labels(node) as labels
                ORDER BY node_id
                """
                
                try:
                    result = session.run(query_nodes, company_id=company_id).data()
                    
                    if not result:
                        print(f"  No {class_name} nodes found within 1 hops")
                        continue
                    
                    print(f"  Found {len(result)} unique {class_name} nodes within 1 hops")
                    
                    # Calculate similarity with CLAIM
                    node_similarities = []
                    for record in result:
                        node = record['node']
                        node_id = record['node_id']
                        
                        # Create text representation
                        node_dict = dict(node)
                        sorted_props = sorted(node_dict.items())
                        
                        node_text_parts = [f"{class_name}:"]
                        for key, value in sorted_props:
                            if value is not None and key not in ['valid_from', 'valid_to', 'is_current', 'embedding']:
                                node_text_parts.append(f"{key}: {value}")
                        node_text = " | ".join(node_text_parts)
                        
                        graph_node_embedding = encoder.encode(node_text, normalize_embeddings=True)
                        similarity = cosine_similarity(claim_embedding, graph_node_embedding)
                        
                        if similarity >= similarity_threshold:
                            node_similarities.append({
                                'node_id': node_id,
                                'node': node,
                                'labels': record['labels'],
                                'similarity': similarity
                            })
                    
                    # Sort and take top-k
                    node_similarities.sort(key=lambda x: (-x['similarity'], x['node_id']))
                    top_nodes = node_similarities[:top_k_per_class]
                    
                    print(f"  Selected {len(top_nodes)} most similar to claim (out of {len(node_similarities)} above threshold)")
                    
                    # STAGE 2: Fetch paths for top-k nodes
                    for item in top_nodes:
                        query_paths = f"""
                        MATCH (company:Organization)
                        WHERE id(company) = $company_id
                        MATCH (node:{class_name})
                        WHERE id(node) = $node_id
                        MATCH path = shortestPath((company)-[*1..1]-(node))
                        WITH path,
                             [rel in relationships(path) | type(rel)] as rel_types,
                             [n in nodes(path) | {{id: id(n), labels: labels(n), properties: properties(n)}}] as path_nodes
                        RETURN 
                            path_nodes,
                            rel_types,
                            length(path) as path_length
                        LIMIT 1
                        """
                        
                        path_result = session.run(query_paths, company_id=company_id, node_id=item['node_id']).data()
                        
                        if path_result:
                            evidence_dict = {
                                'node_id': item['node_id'],
                                'labels': sorted(item['labels']),
                                'properties': dict(item['node']),
                                'similarity': item['similarity'],
                                'embedding_source': 'similar to claim (ALWAYS_INCLUDE)',
                                'paths': [{
                                    'path_nodes': path_result[0]['path_nodes'],
                                    'rel_types': path_result[0]['rel_types'],
                                    'path_length': path_result[0]['path_length']
                                }]
                            }
                            evidence_paths.append(evidence_dict)
                            
                            node_name = dict(item['node']).get('name',
                                           dict(item['node']).get('description',
                                           dict(item['node']).get('target', 'unnamed')))
                            print(f"    {class_name}: {str(node_name)[:50]}... "
                                  f"(similarity={item['similarity']:.3f}, "
                                  f"path_length={path_result[0]['path_length']} hops)")
                    
                except Exception as e:
                    print(f"  Error querying Neo4j for {class_name}: {e}")
                    continue
    
    print(f"\nTotal evidence nodes retrieved: {len(evidence_paths)}")
    
    # Final sort for determinism
    evidence_paths.sort(key=lambda x: (
        -x['similarity'],
        x['node_id'] if x['node_id'] is not None else 0,
        x['labels'][0] if x['labels'] else ''
    ))
    top_k_per_class=temp
    return evidence_paths

def retrieve_evidence(
    company_id: int,
    claim: str,
    claim_idx: int = 0,
    top_k_per_class: int = 10,
    similarity_threshold: float = 0.3,
    embeddings_dir: str = "big_dataset"
) -> List[Dict[str, Any]]:
    """
    Retrieve evidence by:
    1. Load nodes with embeddings from claim_{i}.json
    2. For each JSON node, find 10 most similar nodes in Neo4j graph (same class, connected to company)
    3. For ALWAYS_INCLUDE_CLASSES not in JSON, find 10 most similar to the claim in graph
    4. Return full Neo4j nodes with all attributes
    
    Args:
        company_id: The Neo4j node ID of the company
        claim: The claim text
        year: The year for temporal filtering
        claim_idx: The claim index to load nodes from
        top_k_per_class: Number of most similar nodes to retrieve per JSON node/class
        similarity_threshold: Minimum cosine similarity threshold
        embeddings_dir: Directory containing claim_{i}.json files
    
    Returns:
        List of evidence dictionaries with full node information from Neo4j (deterministically ordered)
    """
    print(f"\n=== Retrieving evidence for company_id={company_id}, claim_idx={claim_idx} ===")
    
    # Load nodes with embeddings from JSON
    json_nodes = load_claim_nodes(claim_idx, embeddings_dir)
    
    if not json_nodes:
        print(f"No nodes found in claim_{claim_idx}.json")
        json_nodes = []
    else:
        print(f"Loaded {len(json_nodes)} nodes with embeddings from claim_{claim_idx}.json")
        
        # Sort JSON nodes deterministically by class, then by a stable property
        json_nodes = sorted(json_nodes, key=lambda n: (
            n.get('class', 'Unknown'),
            str(n.get('properties', {}).get('name', '')),
            str(n.get('properties', {}).get('description', '')),
            str(n.get('properties', {}))
        ))
    
    # Track which classes we've seen in JSON
    json_classes = set()
    evidence = []
    
    with driver.session() as session:
        # Process each JSON node - find similar nodes in graph
        for idx, json_node in enumerate(json_nodes):
            node_class = json_node.get('class', 'Unknown')
            
            json_classes.add(node_class)
            
            # Get the embedding from JSON node
            json_embedding = np.array(json_node['embedding'], dtype=np.float32)
            
            print(f"\n[{idx+1}/{len(json_nodes)}] Processing JSON node of class: {node_class}")
            
            # Query Neo4j: get ALL nodes of this class connected to company
            # ORDER BY for deterministic results
            query = f"""
            MATCH (company:Organization)-[r]-(node:{node_class})
            WHERE id(company) = $company_id
            RETURN 
                id(node) as node_id,
                node,
                type(r) as rel_type,
                labels(node) as labels
            ORDER BY node_id
            """
            
            try:
                result = session.run(query, company_id=company_id).data()
                
                if not result:
                    print(f"  No {node_class} nodes found in graph connected to company")
                    continue
                
                print(f"  Found {len(result)} {node_class} nodes in graph, calculating similarities...")
                
                # Calculate similarity between JSON embedding and each graph node
                node_similarities = []
                
                for graph_node in result:
                    node = graph_node['node']
                    node_id = graph_node['node_id']
                    
                    # Create text representation of graph node for embedding
                    # Sort properties for deterministic text representation
                    node_dict = dict(node)
                    sorted_props = sorted(node_dict.items())
                    
                    node_text_parts = [f"{node_class}:"]
                    for key, value in sorted_props:
                        if value is not None and key not in ['valid_from', 'valid_to', 'is_current', 'embedding']:
                            node_text_parts.append(f"{key}: {value}")
                    node_text = " | ".join(node_text_parts)
                    
                    # Generate embedding for this graph node
                    graph_node_embedding = encoder.encode(node_text, normalize_embeddings=True)
                    
                    # Calculate similarity with JSON node embedding
                    similarity = cosine_similarity(json_embedding, graph_node_embedding)
                    
                    if similarity >= similarity_threshold:
                        node_similarities.append({
                            'node_id': node_id,
                            'node': node,
                            'labels': graph_node['labels'],
                            'rel_type': graph_node['rel_type'],
                            'similarity': similarity
                        })
                
                # Stable sort: by similarity (descending), then by node_id (ascending) for determinism
                node_similarities.sort(key=lambda x: (-x['similarity'], x['node_id']))
                top_nodes = node_similarities[:top_k_per_class]
                
                print(f"  Selected {len(top_nodes)} most similar nodes (out of {len(node_similarities)} above threshold)")
                
                # Add to evidence
                for item in top_nodes:
                    evidence_dict = {
                        'node_id': item['node_id'],
                        'labels': sorted(item['labels']),  # Sort labels for determinism
                        'properties': dict(item['node']),
                        'rel_type': item['rel_type'],
                        'similarity': item['similarity'],
                        'embedding_source': f'similar to JSON node #{idx+1}'
                    }
                    evidence.append(evidence_dict)
                    
                    node_name = dict(item['node']).get('name', dict(item['node']).get('description', dict(item['node']).get('target', 'unnamed')))
                    print(f"    {node_class}: {str(node_name)[:50]}... (similarity={item['similarity']:.3f})")
                
            except Exception as e:
                print(f"  Error querying Neo4j for {node_class}: {e}")
                continue
        
        # Handle ALWAYS_INCLUDE_CLASSES not in JSON
        # Sort for deterministic processing order
        missing_always_include = sorted([cls for cls in ALWAYS_INCLUDE_CLASSES if cls not in json_classes])
        
        if missing_always_include:
            print(f"\n=== Processing ALWAYS_INCLUDE_CLASSES not in JSON: {missing_always_include} ===")
            
            # Generate claim embedding for these classes
            claim_embedding = encoder.encode(claim, normalize_embeddings=True)
            temp = top_k_per_class
            for class_name in missing_always_include:
                if class_name == "KPIObservation":
                    top_k_per_class=3
                else:
                    top_k_per_class=temp

                print(f"\nProcessing ALWAYS_INCLUDE class: {class_name}")
                
                # Query Neo4j for all nodes of this class connected to company
                # ORDER BY for deterministic results
                query = f"""
                MATCH (company:Organization)-[r]-(node:{class_name})
                WHERE id(company) = $company_id
                RETURN 
                    id(node) as node_id,
                    node,
                    type(r) as rel_type,
                    labels(node) as labels
                ORDER BY node_id
                """
                
                try:
                    result = session.run(query, company_id=company_id).data()
                    
                    if not result:
                        print(f"  No {class_name} nodes found in graph")
                        continue
                    
                    print(f"  Found {len(result)} {class_name} nodes in graph")
                    
                    # Calculate similarity with claim
                    node_similarities = []
                    
                    for graph_node in result:
                        node = graph_node['node']
                        node_id = graph_node['node_id']
                        
                        # Create text representation with sorted properties
                        node_dict = dict(node)
                        sorted_props = sorted(node_dict.items())
                        
                        node_text_parts = [f"{class_name}:"]
                        for key, value in sorted_props:
                            if value is not None and key not in ['valid_from', 'valid_to', 'is_current', 'embedding']:
                                node_text_parts.append(f"{key}: {value}")
                        node_text = " | ".join(node_text_parts)
                        
                        # Generate embedding
                        graph_node_embedding = encoder.encode(node_text, normalize_embeddings=True)
                        
                        # Calculate similarity with CLAIM
                        similarity = cosine_similarity(claim_embedding, graph_node_embedding)
                        
                        if similarity >= similarity_threshold:
                            node_similarities.append({
                                'node_id': node_id,
                                'node': node,
                                'labels': graph_node['labels'],
                                'rel_type': graph_node['rel_type'],
                                'similarity': similarity
                            })
                    
                    # Stable sort: by similarity (descending), then by node_id (ascending)
                    node_similarities.sort(key=lambda x: (-x['similarity'], x['node_id']))
                    top_nodes = node_similarities[:top_k_per_class]
                    
                    print(f"  Selected {len(top_nodes)} most similar to claim (out of {len(node_similarities)} above threshold)")
                    
                    # Add to evidence
                    for item in top_nodes:
                        evidence_dict = {
                            'node_id': item['node_id'],
                            'labels': sorted(item['labels']),  # Sort labels for determinism
                            'properties': dict(item['node']),
                            'rel_type': item['rel_type'],
                            'similarity': item['similarity'],
                            'embedding_source': 'similar to claim (ALWAYS_INCLUDE)'
                        }
                        evidence.append(evidence_dict)
                        
                        node_name = dict(item['node']).get('name', dict(item['node']).get('description', dict(item['node']).get('target', 'unnamed')))
                        print(f"    {class_name}: {str(node_name)[:50]}... (similarity={item['similarity']:.3f})")
                    
                except Exception as e:
                    print(f"  Error querying Neo4j for {class_name}: {e}")
                    continue
    
    print(f"\nTotal evidence nodes retrieved from Neo4j: {len(evidence)}")
    
    # Final sort of evidence for deterministic output
    # Sort by: similarity (desc), then node_id (asc), then class (asc)
    evidence.sort(key=lambda x: (
        -x['similarity'],
        x['node_id'] if x['node_id'] is not None else 0,
        x['labels'][0] if x['labels'] else ''
    ))
    top_k_per_class = temp
    return evidence
    

def classify_zero(evidence, company: str, claim: str) -> Tuple[Dict[str, Any], str]:
    """
    Classify a claim using the LLM with retrieved evidence.
    """
    ctx1 = format_entities_as_json(evidence, company) if evidence else json.dumps({
        "company": company,
        "entities": []
    }, indent=2)
    ctx=ctx1

    prompt = ZERO_TMPL.format(company=company, claim=claim, context=ctx)
    
    # Format for Gemma chat template
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant that classifies claims."}]
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt}]
        }
    ]
    
    try:
        # Apply chat template and tokenize
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(model.device, dtype=torch.bfloat16)
        
        input_len = inputs["input_ids"].shape[-1]
        
        # Generate response
        with torch.inference_mode():
            generation = model.generate(
                **inputs,
                max_new_tokens=2000,  
                do_sample=False,     
                temperature=0.0,     
            )
        
        # Decode only the generated part
        generation = generation[0][input_len:]
        response = processor.decode(generation, skip_special_tokens=True)
        
        return _safe_json(response), ctx, response
        
    except Exception as e:
        print(f"Error calling Gemma: {e}")
        return {
            "label": "error_llm_call",
            "reasoning": f"LLM call failed: {e}",
            "cited_node_ids": [],
        }, ctx, response
    
def classify_few(evidence, company: str, claim: str) -> Tuple[Dict[str, Any], str]:
    """
    Classify a claim using the LLM with retrieved evidence.
    """
    ctx = format_entities_as_json(evidence, company) if evidence else json.dumps({
        "company": company,
        "entities": []
    }, indent=2)

    prompt = FEW_TMPL.format(company=company, claim=claim, context=ctx)
    # Format for Gemma chat template
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant that classifies claims."}]
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt}]
        }
    ]
    
    try:
        # Apply chat template and tokenize
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(model.device, dtype=torch.bfloat16)
        
        input_len = inputs["input_ids"].shape[-1]
        
        # Generate response
        with torch.inference_mode():
            generation = model.generate(
                **inputs,
                max_new_tokens=2000,  
                do_sample=False,     
                temperature=0.0,     
            )
        
        # Decode only the generated part
        generation = generation[0][input_len:]
        response = processor.decode(generation, skip_special_tokens=True)
        print(response)
        # print 10 newlines
        print("\n" * 10)
        return _safe_json(response), ctx, response
        
    except Exception as e:
        print(f"Error calling Gemma: {e}")
        return {
            "label": "error_llm_call",
            "reasoning": f"LLM call failed: {e}",
            "cited_node_ids": [],
        }, ctx, response


def classify_zero_python(evidence, company: str, claim: str) -> Tuple[Dict[str, Any], str]:
    """
    Classify a claim using the LLM with retrieved evidence.
    """
    ctx1 = format_entities_as_json(evidence, company) if evidence else json.dumps({
        "company": company,
        "entities": []
    }, indent=2)
    ctx=evidence_to_python_code_inline(ctx1)

    prompt = ZERO_TMPL.format(company=company, claim=claim, context=ctx)
    
    # Format for Gemma chat template
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant that classifies claims."}]
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt}]
        }
    ]
    
    try:
        # Apply chat template and tokenize
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(model.device, dtype=torch.bfloat16)
        
        input_len = inputs["input_ids"].shape[-1]
        
        # Generate response
        with torch.inference_mode():
            generation = model.generate(
                **inputs,
                max_new_tokens=2000,  
                do_sample=False,     
                temperature=0.0,     
            )
        
        # Decode only the generated part
        generation = generation[0][input_len:]
        response = processor.decode(generation, skip_special_tokens=True)
        
        return _safe_json(response), ctx, response
        
    except Exception as e:
        print(f"Error calling Gemma: {e}")
        return {
            "label": "error_llm_call",
            "reasoning": f"LLM call failed: {e}",
            "cited_node_ids": [],
        }, ctx, response
    
def classify_few_python(evidence, company: str, claim: str) -> Tuple[Dict[str, Any], str]:
    """
    Classify a claim using the LLM with retrieved evidence.
    """
    ctx1 = format_entities_as_json(evidence, company) if evidence else json.dumps({
        "company": company,
        "entities": []
    }, indent=2)
    ctx=evidence_to_python_code_inline(ctx1)
    prompt = FEW_TMPL.format(company=company, claim=claim, context=ctx)
    # Format for Gemma chat template
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant that classifies claims."}]
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt}]
        }
    ]
    
    try:
        # Apply chat template and tokenize
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(model.device, dtype=torch.bfloat16)
        
        input_len = inputs["input_ids"].shape[-1]
        
        # Generate response
        with torch.inference_mode():
            generation = model.generate(
                **inputs,
                max_new_tokens=2000,  
                do_sample=False,     
                temperature=0.0,     
            )
        
        # Decode only the generated part
        generation = generation[0][input_len:]
        response = processor.decode(generation, skip_special_tokens=True)
        print(response)
        # print 10 newlines
        print("\n" * 10)
        return _safe_json(response), ctx, response
        
    except Exception as e:
        print(f"Error calling Gemma: {e}")
        return {
            "label": "error_llm_call",
            "reasoning": f"LLM call failed: {e}",
            "cited_node_ids": [],
        }, ctx, response


import json
from typing import Any, Dict

def _create_node_identifier(node: Dict[str, Any]) -> str:
    """
    Create a unique identifier for a node using its type and first property.
    Escapes single quotes for use in Python string literals.
    """
    node_type = node.get("type", "Node")
    properties = node.get("properties", {})
    
    if not properties:
        return node_type
    
    # Get first property value (sorted for consistency)
    for k, v in sorted(properties.items()):
        if v is not None and str(v).strip():  # Check for non-empty values
            str_v = str(v)
            if len(str_v) > 50:
                str_v = str_v[:47] + "..."
            # Escape single quotes for Python strings
            str_v = str_v.replace("'", "\\'")
            return f"{node_type}({str_v})"
    
    return node_type

def evidence_to_python_code_inline(evidence_json: str) -> str:
    """
    Convert evidence to Python code where nodes are instantiated as dictionaries
    before being used in relationships.
    
    Args:
        evidence_json: JSON string from format_entities_as_json()
    
    Returns:
        Python code with node instantiations and relationships
    """
    data = json.loads(evidence_json)
    company_name = data.get("company", "")
    evidence_by_type = data.get("evidence_by_type", {})
    
    code_lines = []
    
    # Add the class definition
    code_lines.append("# Knowledge Base with node instantiation")
    code_lines.append("class KnowledgeBase:")
    code_lines.append("    def __init__(self):")
    code_lines.append("        self.facts = {}")
    code_lines.append("        self.nodes = {}  # Store node objects")
    code_lines.append("    ")
    code_lines.append("    def add_node(self, node_id, node_type, properties):")
    code_lines.append("        \"\"\"Store a node object.\"\"\"")
    code_lines.append("        self.nodes[node_id] = {'type': node_type, 'properties': properties}")
    code_lines.append("    ")
    code_lines.append("    def add_fact(self, entity1, relation, entity2):")
    code_lines.append("        self.facts[(entity1, relation)] = entity2")
    code_lines.append("    ")
    code_lines.append("    def infer(self, entity, *relations):")
    code_lines.append("        current_entity = entity")
    code_lines.append("        for relation in relations:")
    code_lines.append("            key = (current_entity, relation)")
    code_lines.append("            if key in self.facts:")
    code_lines.append("                current_entity = self.facts[key]")
    code_lines.append("            else:")
    code_lines.append("                return None")
    code_lines.append("        return current_entity")
    code_lines.append("    ")
    code_lines.append("    def get_node(self, node_id):")
    code_lines.append("        \"\"\"Get the node object with all its properties.\"\"\"")
    code_lines.append("        return self.nodes.get(node_id)")
    code_lines.append("")
    
    code_lines.append("kb = KnowledgeBase()")
    code_lines.append("")
    
    # Track seen facts and nodes
    seen_facts = set()
    seen_nodes = set()
    
    code_lines.append("# Node instantiations with complete properties")
    
    # First pass: collect all unique nodes
    all_nodes = []
    for entity_type, entities in sorted(evidence_by_type.items()):
        for entity in entities:
            connection = entity.get("connection")
            if connection is None:
                continue
            
            nodes = connection.get("nodes", [])
            for node in nodes:
                node_id = _create_node_identifier(node)
                if node_id not in seen_nodes:
                    seen_nodes.add(node_id)
                    all_nodes.append({
                        'id': node_id,
                        'type': node.get("type", "Node"),
                        'properties': node.get("properties", {})
                    })
    
    # Group nodes by type for organization
    nodes_by_type = {}
    for node in all_nodes:
        node_type = node['type']
        if node_type not in nodes_by_type:
            nodes_by_type[node_type] = []
        nodes_by_type[node_type].append(node)
    
    # Output node instantiations grouped by type
    for node_type in sorted(nodes_by_type.keys()):
        code_lines.append(f"\n# {node_type} nodes")
        for node in nodes_by_type[node_type]:
            node_id = node['id']
            props = node['properties']
            props_repr = repr(props)
            
            code_lines.append(f"kb.add_node(")
            code_lines.append(f"    node_id='{node_id}',")
            code_lines.append(f"    node_type='{node_type}',")
            code_lines.append(f"    properties={props_repr}")
            code_lines.append(f")")
    
    code_lines.append("")
    code_lines.append("# Relationships (facts) between nodes")
    
    # Reset seen facts
    seen_facts = set()
    
    for entity_type, entities in sorted(evidence_by_type.items()):
        has_facts = False
        
        for entity in entities:
            connection = entity.get("connection")
            
            if connection is None:
                continue
            
            nodes = connection.get("nodes", [])
            relationships = connection.get("relationships", [])
            summary = connection.get("summary", "")
            
            if not nodes:
                continue
            
            # Add section header if this is the first fact for this type
            if not has_facts:
                code_lines.append(f"\n# {entity_type} relationships")
                has_facts = True
            
            # Add path summary as comment
            code_lines.append(f"# Path: {summary}")
            
            # Add the facts
            for i in range(len(nodes) - 1):
                node1 = nodes[i]
                node2 = nodes[i + 1]
                relation = relationships[i] if i < len(relationships) else "CONNECTED_TO"
                
                node1_id = _create_node_identifier(node1)
                node2_id = _create_node_identifier(node2)
                
                fact_key = (node1_id, relation, node2_id)
                
                if fact_key not in seen_facts:
                    seen_facts.add(fact_key)
                    code_lines.append(f"kb.add_fact('{node1_id}', '{relation}', '{node2_id}')")
    
    return "\n".join(code_lines)