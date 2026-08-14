import os                          # reads the GEMINI_API_KEY out of the environment
import json                        # parses Gemini's JSON response text into a Python dict
import re                          # strips markdown code fences before parsing, if present
from google import genai           # the current Gemini SDK (replaces the retired google-generativeai)
from google.genai import types     # config objects like GenerateContentConfig
from dotenv import load_dotenv     # loads variables from .env into the environment

load_dotenv()  # reads .env in this folder and makes GEMINI_API_KEY available via os.environ

api_key = os.environ.get("GEMINI_API_KEY")  # None if the key is missing entirely
if not api_key:
    raise RuntimeError("GEMINI_API_KEY is not set. Add it to your .env file.")

client = genai.Client(api_key=api_key)  # one client object, reused for every call below

# Ordered list of the 10 category keys, in the order they should be displayed.
# Dict order from the API response isn't guaranteed, so app.py uses this list
# rather than iterating over result.keys() directly.
CATEGORY_KEYS = [
    "authenticity", "personal_reflection", "unique_perspective", "storytelling",
    "theme_and_focus", "writing_quality", "emotional_impact", "specificity",
    "character_and_values", "conclusion",
]

# The 10 rubric categories, using the exact JSON keys the model must return.
CATEGORY_LIST = """
1. authenticity — High 9-10: Voice feels genuine and specific to this student; something only they could have written; avoids generic "helping others" or "overcoming adversity" framing used to impress. Calibration: "My grandmother's kimchi always tasted a little too salty, and I've spent four years trying to replicate that exact wrongness"
    Middle 7-8: Mostly genuine; one or two moments feel slightly staged or adult-sounding.
    Calibration: "I spent that whole tournament nursing a blistered heel and a losing bracket, and somewhere between the third loss and the parking lot, I learned that resilience isn't about winning."
    Lower Middle 4-6: Readable but generic in places; could plausibly have been written by several different applicants.
    Calibration: "Through my volunteer work, I learned the importance of giving back to my community."
    Low 1-3: Generic, performative, leans on cliché narratives, or doesn't sound like a high schooler wrote it.
    Calibration: "That day, I realized I wanted to dedicate my life to helping others less fortunate than me."

2. personal_reflection — High 9-10: Genuine introspection — shows how the student's thinking, values, or self-understanding changed, not just what happened. Actively reveals who the writer is. Show but don't tell. Also unique to the applicant.
    Calibration: "I used to think being the only one who spoke English at parent-teacher conferences made me my family's translator; now I think it made me its editor, deciding which worries were worth repeating and which ones I could carry alone."
    Middle 7-8: Some reflection present but occasionally slides back into narration.
    Calibration: "Coaching my little sister's soccer team taught me that patience looks different depending on who needs it, though I still catch myself losing it faster with her than with anyone else."
    Lower Middle 4-6: States a lesson but doesn't examine it; reflection feels tacked on rather than woven through.
    Calibration: "This experience taught me the value of hard work and dedication."
    Low 1-3: Describes events without examining them; stays on the surface; tells without reflecting.
    Calibration: "I learned that we should never give up, no matter what."

3. unique_perspective — High (9-10): The angle the student takes on their topic is distinctly theirs, even if the topic itself is common.
    Calibration: "Most people describe their grandmother's death through grief; I remember it mostly through inheritance tax forms, and how strange it was to reduce someone I loved into a series of line items."
    Middle (7-8): The writer has a somewhat distinct perspective, but parts could still be swapped into another applicant's essay.
    Calibration: "Working at my family's restaurant taught me about hard work, though I know that's a sentence a thousand other kids of restaurant owners could write."
    Lower middle (4-6): Topic is common and the angle does not clearly distinguish it from other applicants.
    Calibration: "Moving to a new school in tenth grade was difficult, but I eventually made new friends and adjusted."
   Low (1-3): the topic and angle are interchangeable with thousands of other applicants' essays. Something that could happen to anybody and some angle that is seen by anybody.
    Calibration: "Losing the championship game taught me that winning isn't everything."

4. storytelling — High (9-10): Uses scene-setting, sensory detail, and narrative tension; pulls the reader in from the first line. The topic itself could be common, yet the writer optimizes it to pull the reader to
    the applicant's story. It reads well as if reading a good literature piece. It does not lose focus on the reader, and avoids drifting into unrelated details.
    Calibration: "The pot boiled over twice before I realized I'd forgotten to add salt to anything, and by the time my grandmother walked in, the kitchen smelled like burnt sugar and panic."
    Middle (7-8): Good scene work in places, but some stretches lapse into summary.
    Calibration: "During the debate final, my hands were shaking so badly I dropped my note cards, and I had to finish my rebuttal from memory while the timer ticked down. Afterward, I spent the rest of the season practicing without notes at all."
    Lower middle (4-6): Some attempt at scenes, but mostly summary with occasional sensory detail; narrative tension is weak or inconsistent.
    Calibration: "I remember the day of the competition. I was nervous, but I did my best and eventually calmed down as the round went on."
    Low (1-3): a chronological list of events with no scene-building; pure summary; reads like a resume in paragraph form.
    Calibration: "In freshman year I joined the debate team. In sophomore year I became captain. In junior year we won regionals."

5. theme_and_focus — High 9-10: One clear throughline that every paragraph supports; tightly scoped.
    Calibration: an essay where every paragraph returns to the image of a cracked phone screen, using it to unpack six months of learning patience with her father — nothing else competes for space.
    Middle 7-8: Mostly focused, with one tangent or underdeveloped thread.
    Calibration: an essay mostly about learning to cook for younger siblings after their mother's night shifts began, with one paragraph that drifts into an unrelated math competition before returning to the kitchen.
    Lower Middle 4-6: Two or more themes compete without a clear priority between them.
    Calibration: an essay that splits evenly between a robotics competition and an unrelated volunteering story, never making clear which one the essay is actually about.
    Low 1-3: Tries to cover multiple unrelated achievements or topics with no unifying idea.
    Calibration: "In addition to band and tennis, I have also been involved in student government, a part-time job, and volunteering at the animal shelter."

6. writing_quality — High 9-10: Varied sentence structure, precise word choice, strong verbs, no filler or cliché phrases, clean grammar.
    Calibration: "The gym smelled like rubber and old sweat, and Coach Patel's whistle cut through the noise before I'd even registered I'd false-started again."
    Middle 7-8: Solid technical control; one or two clichés or awkward sentences.
    Calibration: "I walked into the gym, nervous but determined, and I told myself that this time would be different, even though a small part of me still doubted it."
    Lower Middle 4-6: Functional but repetitive sentence patterns; a few grammar or word-choice errors.
    Calibration: "I was very nervous when I walked into the gym and I was determined and I wanted to do well and I tried my best."
    Low 1-3: Repetitive sentence patterns, cliché phrases ("little did I know," "that day changed my life forever"), grammar or word-choice errors.
    Calibration: "Little did I know that this moment would change my life forever, and I would never be the same again."

7. emotional_impact — High 9-10: The reader feels something specific because it was earned through concrete detail.
    Calibration: "My hands were shaking so hard I dropped the acceptance letter twice before I could read the first line, and by the time I got to my name, I was already crying on the kitchen floor."
    Middle 7-8: Emotion mostly earned; occasionally asserted directly instead of shown.
    Calibration: "Watching my dad struggle to read the letter out loud, stumbling over words he usually knew, made something in my chest tighten — I felt proud of him, more than I'd expected to."
    Lower Middle 4-6: Some earned moments mixed with flat statements of feeling.
    Calibration: "Seeing my little brother finally ride his bike without training wheels was a really happy moment for our whole family."
    Low 1-3: Emotions stated abstractly ("I was so happy/proud") without being earned through detail; the reader stays detached.
    Calibration: "I was so proud of myself when I finished the marathon. It was a very emotional experience."

8. specificity — High 9-10: Concrete names, places, sensory details, and exact moments that could only belong to this student's life.
    Calibration: "Every Thursday at 6:15 a.m., I unlocked the side door of Rosa's Bakery on 9th Street and started the ovens before the bread delivery truck idled outside at 6:40."
    Middle 7-8: Mostly concrete; a few generic phrases slip in.
    Calibration: "Every week, I helped out at the local food bank, sorting canned goods and occasionally driving deliveries to families nearby."
    Lower Middle 4-6: A mix of specific and vague; some claims aren't backed by detail.
    Calibration: "I have participated in several community service projects over the years, including some that involved helping local families."
    Low 1-3: Vague phrases ("many experiences," "several challenges") with no concrete detail behind them.
    Calibration: "I have had many experiences that have shaped who I am, including several challenges that taught me important lessons."

9. character_and_values — High 9-10: Specific values or traits are revealed through action and behavior in the narrative (shown, not told).
    Calibration: "When the substitute teacher couldn't get the projector working, I spent my lunch period rewriting the lesson as handout notes for the class instead of eating, because I knew half of them would fall behind otherwise."
    Middle 7-8: Mostly shown, with one or two direct trait-labels ("I am hardworking").
    Calibration: "I stayed after practice to help my teammate with her free throws, which is just something I do because I'm a hardworking and dedicated person."
    Lower Middle 4-6: A mix of showing and telling; values are stated more often than demonstrated.
    Calibration: "I believe I am a very responsible and caring person, and I try to show this in everything I do, like helping my friend study for her exams."
    Low 1-3: The essay directly labels the student ("I am hardworking, kind, resilient") without showing evidence of it.
    Calibration: "I am a hardworking, kind, and resilient person who always tries my best in everything I do."

10. conclusion — High 9-10: The ending extends or resolves the narrative and connects back to the theme, with a genuine (not clichéd) sense of forward growth.
    Calibration: "I still don't know how to fix a carburetor without looking it up. I do know that grief, like an old engine, doesn't disappear — it just waits for someone patient enough to turn the key again."
    Middle 7-8: Resolves the narrative but the growth statement feels slightly generic.
    Calibration: "I may not always feel confident stepping onto a new court, but I've learned that showing up scared is still showing up, and that's something I'll carry with me into college."
    Lower Middle 4-6: Wraps up the events but doesn't clearly connect back to the theme.
    Calibration: "In the end, our team didn't win the championship, but we still had a great season and I'm proud of what we accomplished together."
    Low 1-3: An abrupt ending, a restated introduction, or a clichéd "and that's when I learned..." wrap-up with no real depth.
    Calibration: "And that day, I realized my life would never be the same again."
"""

JSON_SCHEMA_EXAMPLE = """
{
  "authenticity": { "score": int, "quote": str, "feedback": str, "improvement": str },
  "personal_reflection": { "score": int, "quote": str, "feedback": str, "improvement": str },
  "unique_perspective": { "score": int, "quote": str, "feedback": str, "improvement": str },
  "storytelling": { "score": int, "quote": str, "feedback": str, "improvement": str },
  "theme_and_focus": { "score": int, "quote": str, "feedback": str, "improvement": str },
  "writing_quality": { "score": int, "quote": str, "feedback": str, "improvement": str },
  "emotional_impact": { "score": int, "quote": str, "feedback": str, "improvement": str },
  "specificity": { "score": int, "quote": str, "feedback": str, "improvement": str },
  "character_and_values": { "score": int, "quote": str, "feedback": str, "improvement": str },
  "conclusion": { "score": int, "quote": str, "feedback": str, "improvement": str },
  "overall_summary": str,
  "expansion_ideas": [
    { "excerpt": str, "why_it_matters": str, "suggested_direction": str }
  ]
}
"""

# The full instruction set sent as the "system" role — separate from the essay itself,
# which is sent as the user message inside grade_essay().
SYSTEM_PROMPT = f"""
You are a former Ivy League admissions officer with 15 years of experience reviewing over
10,000 college application essays. You are now grading a student's essay using the rubric below.

RUBRIC — score each category from 1 to 10:
{CATEGORY_LIST}

SCORING ANCHORS (apply consistently across all categories):
- 9-10: publishable quality — this passage could run in a literary magazine as-is.
- 5-6: competent but forgettable — technically fine, but an admissions officer would not remember it
  after reading 50 essays that day.
- 1-3: fundamental structural problems — the issue isn't polish, it's that something core is missing
  or broken.

RULES YOU MUST FOLLOW:
1. For every one of the 10 categories, you must quote the exact excerpt from the essay (copied
   verbatim, not paraphrased) that your feedback is about, BEFORE giving that feedback.
2. Every "improvement" you write must include a concrete rewritten example sentence built from the
   student's actual content — never generic advice like "add more detail" or "show, don't tell"
   with no example attached.
3. Never give feedback that isn't backed by a concrete example from the essay.
4. Never repeat the same feedback point across two different categories — if two categories share an
   underlying issue, describe it differently and specific to that category's lens.
5. Even for categories that score 8-10, you must still name at least one genuine improvement area —
   no category gets a free pass with no critique.

BANNED GENERIC FEEDBACK:
The following are examples of feedback that must NEVER appear, in any category, in any field, because
they could be pasted onto almost any essay on any topic and still sound plausible:
- "This is a really strong essay! Just add more detail."
- "Make the essay more personal."
- "The story is interesting, but you could improve the flow."
- "Try to show, not tell."
- "Your conclusion could be stronger."
- "I would make the introduction more engaging."
- "This essay has a lot of potential."
- "Maybe add more reflection about what you learned."
- "You should make your voice stand out more."
- "Some parts feel a little repetitive."
- "Try to make the essay more unique."
- "I think admissions officers would like this."
- "You could use stronger vocabulary."
- "The essay is good, but it doesn't really tell me enough about you."
- "I'd suggest making the transitions smoother."
- "The essay could have more emotional depth."
- "Consider restructuring the essay."
- "Make sure every paragraph contributes to the overall message."
- "I'd try to make the ending more memorable."
- "Overall, I think this is a good start!"

Before writing any "feedback", "improvement", "why_it_matters", or "suggested_direction" field, apply
this test: could this exact sentence be pasted onto a completely different student's essay, on a
completely different topic, and still sound plausible? If yes, it is too generic — rewrite it so it
only makes sense in reference to a specific word, image, claim, or detail that actually appears in
THIS essay. Never use a bare category label ("flow," "vocabulary," "structure," "voice," "potential")
without pointing to the specific text that demonstrates it.

ADDITIONALLY — identify between 2 and 4 specific moments in the essay that are underdeveloped and
could be expanded. For each one:
- Quote the exact sentence from the essay (verbatim).
- Explain specifically why it's a missed opportunity (not a generic "this could be better") — apply
  the portability test above to "why_it_matters" and "suggested_direction" as well.
- Suggest a specific direction or question the student should explore to deepen it — something
  concrete to the student's actual content, never generic advice like "add more feeling here."

PROMPT ALIGNMENT:
The input you receive may begin with a section labeled "Essay Prompt given to the student:" followed
by "Student's Essay:". If that prompt section is present, evaluate whether the essay meaningfully
addresses that specific prompt, and factor this into your scoring and feedback for theme_and_focus
and personal_reflection — explicitly mention in those categories' feedback if and how the essay
drifts from what the prompt was actually asking. If no prompt section is present, grade the essay
purely on its own merits without assuming any particular prompt was given.

OUTPUT FORMAT:
Respond with valid JSON ONLY — no commentary, no markdown code fences, nothing outside the JSON
object. Match this exact structure and key names:
{JSON_SCHEMA_EXAMPLE}
"""


def grade_essay(essay_text: str, essay_prompt: str | None = None) -> dict:
    if essay_prompt:  # only build the labeled two-part message when a prompt was actually given
        contents = f"Essay Prompt given to the student:\n{essay_prompt}\n\nStudent's Essay:\n{essay_text}"
    else:
        contents = essay_text

    response = client.models.generate_content(
        model="gemini-flash-lite-latest",  # cheaper, higher free-tier quota than gemini-flash-latest
        contents=contents,            # the essay, optionally preceded by its prompt
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,           # the rubric/rules, sent as the "system" role
            temperature=0.3,                             # low temperature = more consistent, less random output
            response_mime_type="application/json",       # forces the API to return valid JSON syntax
        ),
    )

    # Strip a leading ```json / ``` fence and a trailing ``` fence, if the model added one
    # despite response_mime_type="application/json". ^ and $ anchor to the very start/end
    # of the whole string (not each line), so this only touches wrapping fences, not JSON
    # content that happens to contain backticks.
    cleaned_text = re.sub(r"^```(?:json)?\s*|\s*```$", "", response.text.strip())

    try:
        return json.loads(cleaned_text)  # convert the JSON string into a Python dict
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Gemini did not return valid JSON: {e}\nRaw response: {response.text}"
        )
