import pandas as pd
import google.generativeai as genai
import os
from dotenv import load_dotenv
import tiktoken
import time
import re
from langchain_groq import ChatGroq
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage

load_dotenv()
current_key_index = 0


claim_gen_with_type4 = """

You are an **ESG (Environmental, Social, Governance) and Greenwashing claim generator.**  

**Task:**  
Given the following article, generate **at least 8 short, plain-language claims** that could realistically appear in one of the following contexts:  
- a company press release  
- a news headline  
- a product advertisement  
- an everyday statement about sustainability  

Each claim should reflect either a **truthful sustainability statement** or a **greenwashing claim**.  
---
Greenwashing Types  
1. **Type 1 - Misleading labels:** Using vague or unreliable sustainability labels.  
2. **Type 2 - Legal obligations as achievements:** Presenting compliance with law as an eco-feature.  
3. **Type 3 - Overgeneralization:** Claiming sustainability for an entire product when it only applies to part/aspect.  
4. **Type 4 - Unsupported claims:** Making environmental claims for which the company cannot provide evidence.
---
Guidelines  
1. Each claim must be **understandable on its own**, without external context.  
   - Include enough detail (company, product, or action).  
   - Example: Instead of "We cut emissions by 50%," write: "Company X claims it cut its carbon emissions by 50% in 2024." , where Company X is the name of the given Company.
2. For each claim, assign a label:  
   - **greenwashing** if it fits one of the types above.  
   - **not_greenwashing** if it is a valid, evidence-based sustainability claim.  
3. Always include the **company name**.  
4. Provide a **short justification** explaining why the claim is labeled as such.  
5. **Balance requirement:** At least **one claim must be labeled greenwashing** and at least **one claim must be labeled not_greenwashing** for every article.  
6. **Variation requirement:** Ensure claims vary in style and tone (at least one press release-like, one news headline-like, one everyday statement or product advertisement).  
7. **Coverage requirement:** Across the greenwashing claims, ensure that at least one example of **Type 1, Type 2, Type 3 and Type 4** is represented whenever possible.  
---
Output Format (strictly follow this)  
CLAIM: <claim text> || LABEL: <greenwashing/not_greenwashing>  
COMPANY: <company name>  
JUSTIFICATION: <brief reasoning for the label>  
---
Example Outputs  
**Example 1 (Type 1 - Misleading label, press release style):**  
CLAIM: H&M promotes its new "Conscious Collection" as a sustainable fashion line. || LABEL: greenwashing  
COMPANY: H&M  
JUSTIFICATION: The "Conscious" label is vague and does not provide clear criteria for sustainability, fitting Type 1 greenwashing.  
**Example 2 (Type 2 - Legal compliance framed as achievement, news headline style):** 
CLAIM: McDonald's announces it is reducing plastic waste by introducing reusable cutlery in France. || LABEL: greenwashing  
COMPANY: McDonald's  
JUSTIFICATION: This is a legal requirement in France since January 2023, so presenting it as a sustainability achievement is misleading.  
**Example 3 (Type 3 - Overgeneralization, product advertisement style):**  
CLAIM: Walmart advertises its "eco-friendly" bamboo towels as fully sustainable. || LABEL: greenwashing  
COMPANY: Walmart  
JUSTIFICATION: The towels are made from rayon derived from bamboo, a process that eliminates environmental benefits, fitting Type 3 greenwashing.
**Example 4 (Type 4 - Unsupported claim, press release style):**  
CLAIM: Keurig states that all of its single-use coffee pods are "easily recyclable in any city recycling program." || LABEL: greenwashing  
COMPANY: Keurig  
JUSTIFICATION: Keurig was fined for misleading claims, as many recycling facilities could not process its pods, making this an unsupported claim and fitting Type 4.
**Example 5 (Not greenwashing, genuine action, everyday statement style):**  
CLAIM: Tesla reports that its Gigafactory in Nevada now runs entirely on renewable energy. || LABEL: not_greenwashing  
COMPANY: Tesla  
JUSTIFICATION: Running a facility fully on renewable energy is a verifiable and substantive sustainability claim.  
---
**Article:**  

"""


# Add your API keys here
API_KEYS = [
    os.getenv("YOUR_API_KEY_1"),
    os.getenv("YOUR_API_KEY_2"),
    os.getenv("YOUR_API_KEY_3"),
]


# Helper functions to extract claims, labels, companies, justifications
def extract_claims(text: str):
    pattern = r"CLAIM:\s*(.*?)\s*\|\|\s*LABEL:\s*(greenwashing|not_greenwashing)"
    matches = re.findall(pattern, text, flags=re.IGNORECASE)
    df = pd.DataFrame(matches, columns=["claim", "label"])
    return df


def extract_claims(text: str):
    return re.findall(r"CLAIM:\s*(.*?)\s*\|\|", text, flags=re.DOTALL)


def extract_labels(text: str):
    return re.findall(r"LABEL:\s*(.*?)\s*COMPANY:", text, flags=re.DOTALL)


def extract_companies(text: str):
    return re.findall(r"COMPANY:\s*(.*?)\s*JUSTIFICATION:", text, flags=re.DOTALL)


def extract_justifications(text: str):
    return re.findall(r"JUSTIFICATION:\s*(.*?)(?=\nCLAIM:|\Z)", text, flags=re.DOTALL)


def extract_all(text: str):
    claims = extract_claims(text)
    labels = extract_labels(text)
    companies = extract_companies(text)
    justifications = extract_justifications(text)

    results = []
    for c, l, co, j in zip(claims, labels, companies, justifications):
        results.append(
            {
                "claim": c.strip(),
                "label": l.strip(),
                "company": co.strip(),
                "justification": j.strip(),
            }
        )
    return results


# Function to approximate token count
def approximate_token_count(text: str, model="gemini-2.0-flash"):

    try:
        tokenizer = tiktoken.get_encoding("cl100k_base")
        tokens = tokenizer.encode(text)
        return len(tokens)
    except Exception as e:
        print(f"Token count approximation failed: {e}")
        return 0


def call_llm(text):

    time.sleep(2)
    global current_key_index
    full_prompt = f"{claim_gen_with_type4}\n\n:\n{text}"

    token_count = approximate_token_count(full_prompt)

    model_name = "gemini-2.5-flash" if token_count < 250_000 else "gemini-2.0-flash"

    total_keys = len(API_KEYS)
    attempts = 0

    while attempts < total_keys:
        key = API_KEYS[current_key_index]
        if not key:
            print(f"[Key #{current_key_index + 1}] API key is missing or empty.")
        else:
            try:
                time.sleep(2)  # Respect rate limits
                genai.configure(api_key=key)
                model = genai.GenerativeModel(model_name)

                response = model.generate_content(
                    full_prompt, generation_config={"temperature": 0.0}
                )
                if response and hasattr(response, "text"):
                    print(f"[Key #{current_key_index + 1}] Success")
                    return response.text.strip(), model_name
            except Exception as e:
                print(f"[Key #{current_key_index + 1}] LLM call failed: {e}")

        current_key_index = (current_key_index + 1) % total_keys
        attempts += 1

    return call_llm2(full_prompt)


# Second LLM call using Groq. You can add your API keys in the .env file.
def call_llm2(full_prompt):

    groq_keys = [os.getenv("YOUR_API_KEY_1"), os.getenv("YOUR_API_KEY_2")]
    model_name = "llama-3.3-70b-versatile"

    for i, key in enumerate(groq_keys, start=1):
        if not key:
            print(f"[ Key #{i}] Missing or empty.")
            continue

        try:
            time.sleep(2)
            llm = ChatGroq(model=model_name, api_key=key, temperature=0.0)
            response = llm.invoke([HumanMessage(content=full_prompt)])
            print("Response:", response.content.strip())

            if response and response.content:
                print(f"[Groq Key #{i}] Success")

                return response.content.strip(), model_name
            else:
                print(f"[Groq Key #{i}] Empty response")

        except Exception as e:
            print(f"[Groq Key #{i}] Error: {e}")

    print("All Groq API keys failed or are exhausted.")
    return call_llm3(full_prompt)


# Third LLM call using Ollama. Make sure you have Ollama set up locally.
# This does not require an API key.
def call_llm3(text):
    model_name = "mistral-small3.2:24b"
    full_prompt = f"{claim_gen_with_type4}\n\n:\n{text}"

    try:
        llm = ChatOllama(model=model_name, temperature=0.0)
        response = llm.invoke([HumanMessage(content=full_prompt)])

        if response and response.content:
            print(f"[Ollama {model_name}] Success")
            return response.content.strip(), model_name
        else:
            print(f"[Ollama {model_name}] Empty response")
    except Exception as e:
        print(f"[Ollama {model_name}] Error: {e}")

    print("All models failed.")
    return None, None


final_df = []
df = pd.read_csv("your_news_data_file_location.csv")  # Update with your data file path
for idx, row in df.iterrows():
    print(f"Processing row {idx + 1}/{len(df)}")

    # Call your LLM
    result, model = call_llm(
        "Title:\n\n"
        + str(row["title"])
        + "Content:\n\n"
        + str(row["content"])
        + "Company Name:\n\n"
        + str(row["Organization_llm"])
    )

    # Extract structured claims using regex
    claims = extract_all(result)  # list of dicts

    if claims:  # only if something extracted
        # Convert to dataframe
        small_df = pd.DataFrame(claims)
        # Add metadata
        small_df["id"] = row["id"]
        small_df["model"] = model
        final_df.append(small_df)


# Concatenate all results
if final_df:
    final_df = pd.concat(final_df, ignore_index=True)
    final_df.to_csv("your_result_dataframe.csv", index=False)
    print(final_df.shape)
else:
    print("No claims were extracted.")
