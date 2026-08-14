import os                          # reads the GEMINI_API_KEY out of the environment
import json                        # parses Gemini's JSON response text into a Python dict
import re                          # strips markdown code fences before parsing, if present
from google import genai           # the Gemini SDK
from google.genai import types     # config objects like GenerateContentConfig
from dotenv import load_dotenv     # loads variables from .env into the environment

load_dotenv()  # reads .env in this folder and makes GEMINI_API_KEY available via os.environ

api_key = os.environ.get("GEMINI_API_KEY")  # None if the key is missing entirely
if not api_key:
    raise RuntimeError("GEMINI_API_KEY is not set. Add it to your .env file.")

client = genai.Client(api_key=api_key)  # separate client instance from grading.py's, same API key

JSON_SCHEMA_EXAMPLE = """
{
  "topics": [
    { "topic": str, "strength_score": int, "strengths": str, "risks": str, "suggested_angle": str }
  ],
  "recommended_topic": str,
  "overall_advice": str
}
"""

SYSTEM_PROMPT = f"""
You are a former Ivy League admissions officer with 15 years of experience reviewing over
10,000 college application essays. A student has NOT written an essay yet — they've listed several
possible topics and want your read on which one is most worth developing.

For each topic submitted, evaluate:
- strength_score (1-10): how promising this is as a foundation for a standout essay.
  9-10: a genuinely distinctive angle waiting to be written. 5-6: workable but generic as stated.
  1-3: a well-worn topic likely to blend into thousands of other applications without a sharp angle.
- strengths: what makes THIS specific topic promising, referencing what the student actually wrote —
  never generic praise like "this could work well" or "this has potential."
- risks: the specific cliché or pitfall this exact kind of topic is prone to (for example: "sports
  injury comeback stories are extremely common; without a distinctive detail this risks reading like
  a highlight reel").
- suggested_angle: one concrete, specific angle or focusing question that would make THIS topic
  distinctive — never generic advice like "make it more personal" or "add more detail."

Before writing any "strengths", "risks", or "suggested_angle" field, apply this test: could this
exact sentence be pasted onto a different topic entirely and still sound plausible? If yes, it's too
generic — rewrite it so it only makes sense for the specific topic the student actually described.

Then compare the topics against each other and set "recommended_topic" to the one with the strongest
potential, and "overall_advice" to a short explanation of specifically why it beats the others — not
a restatement of its individual strengths in isolation.

OUTPUT FORMAT:
Respond with valid JSON ONLY — no commentary, no markdown code fences, nothing outside the JSON
object. Match this exact structure and key names:
{JSON_SCHEMA_EXAMPLE}
"""


def analyze_topics(topics: list[str]) -> dict:
    numbered_topics = "\n".join(f"{i}. {t}" for i, t in enumerate(topics, start=1))  # "1. ...\n2. ..."

    response = client.models.generate_content(
        model="gemini-flash-lite-latest",  # same model grade_essay() uses
        contents=numbered_topics,          # the numbered list of topics, sent as the user message
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.3,                        # low temperature = more consistent, less random output
            response_mime_type="application/json",  # forces the API to return valid JSON syntax
        ),
    )

    # Same fence-stripping safeguard as grade_essay(), in case the model wraps the JSON
    # in a markdown code block despite response_mime_type="application/json".
    cleaned_text = re.sub(r"^```(?:json)?\s*|\s*```$", "", response.text.strip())

    try:
        return json.loads(cleaned_text)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Gemini did not return valid JSON: {e}\nRaw response: {response.text}"
        )
