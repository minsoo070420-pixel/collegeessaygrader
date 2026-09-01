import os                  # reads PORT and FLASK_DEBUG from the environment
from datetime import date  # used to detect when a new calendar day starts, to reset the daily count
from markupsafe import Markup, escape  # builds safe HTML manually, for annotating the essay display
from flask import Flask, render_template, request  # render_template loads HTML files; request reads form data
from grading import grade_essay, DIMENSION_KEYS, DIMENSION_LABELS  # the grading call + dimension metadata
from topic_analysis import analyze_topics as compare_topics  # aliased to avoid clashing with the view function below

app = Flask(__name__)  # creates the Flask application object that routes attach to

MIN_WORD_COUNT = 50   # essays shorter than this are rejected
MAX_WORD_COUNT = 650  # essays longer than this are rejected

DAILY_GEMINI_LIMIT = 100  # shared cap across every route that calls Gemini (essay grading AND topic analysis)
                           # raised from 15 now that the app runs on the cheaper, higher-quota lite model
                           # set to None to remove the app-level cap entirely (Gemini's own quota still applies)

# In-memory counter, shared by every request this process handles — and by every route that
# calls Gemini, not just essay grading. Resets to 0 whenever the calendar date changes. Does NOT
# persist across server restarts, and does NOT stay in sync across multiple worker processes if
# this app is ever deployed with more than one.
_gemini_call_count = 0
_gemini_call_count_date = None

MIN_TOPICS = 2  # fewest topics a user can submit on the /topics page
MAX_TOPICS = 5  # most topics a user can submit on the /topics page

def _daily_limit_reached():
    """Checks and updates the shared daily Gemini-call counter.

    Returns True if today's budget is already used up (caller should NOT call Gemini).
    Otherwise increments the counter — since the caller is about to spend a slot — and
    returns False.
    """
    global _gemini_call_count, _gemini_call_count_date
    today = date.today()
    if _gemini_call_count_date != today:  # first request of a new calendar day
        _gemini_call_count_date = today
        _gemini_call_count = 0

    if DAILY_GEMINI_LIMIT is not None and _gemini_call_count >= DAILY_GEMINI_LIMIT:
        return True

    _gemini_call_count += 1  # counted before the caller's Gemini call, since reaching Gemini uses
                              # a quota slot even if the response later fails to parse
    return False

def annotate_essay(essay_text, dimensions, deeper_ideas):
    """Wraps each quoted excerpt found in essay_text with a numbered, highlighted <mark> tag,
    like a Google-Docs-style comment marker. Returns (annotated_html, comments):

    - annotated_html: a Markup object (pre-escaped, safe to render directly without Jinja
      re-escaping it) with each matched excerpt wrapped in <mark> and a small superscript number.
    - comments: a list of {"number", "label", "text", "css_class"} dicts, in the same
      left-to-right order the numbers appear in the essay, meant to be listed below it.

    Colors alternate by position (1st, 3rd, 5th... = accent; 2nd, 4th, 6th... = primary),
    not by dimension vs. "go deeper" idea type — this makes adjacent comments visually distinct
    from each other, regardless of which kind of feedback they are.

    Quotes that can't be found verbatim in the essay (the AI isn't always perfectly literal
    despite the rules) are simply skipped — this never raises an error. prompt_fit has no
    single quote (it judges the whole essay against the prompt), so it never produces a span.
    """
    # (start, end, label, comment_text) for every quote that's an exact substring of the essay
    spans = []
    for key, data in dimensions:
        quote = data.get("quote", "")
        start = essay_text.find(quote) if quote else -1
        if start != -1:
            label = DIMENSION_LABELS.get(key, key)
            spans.append((start, start + len(quote), label, data.get("feedback", "")))
    for idea in deeper_ideas:
        excerpt = idea.get("excerpt", "")
        start = essay_text.find(excerpt) if excerpt else -1
        if start != -1:
            spans.append((start, start + len(excerpt), "Where You Could Go Deeper", idea.get("why_it_matters", "")))

    spans.sort(key=lambda s: s[0])  # process left-to-right through the essay

    pieces = []
    comments = []
    cursor = 0
    number = 1
    for start, end, label, comment_text in spans:
        if start < cursor:  # overlaps a span already placed — skip rather than produce broken markup
            continue
        css_class = "annotation-accent" if number % 2 == 1 else "annotation-primary"  # alternate by position
        pieces.append(escape(essay_text[cursor:start]))  # plain text before this quote, escaped
        pieces.append(Markup(f'<mark class="{css_class}">'))
        pieces.append(escape(essay_text[start:end]))  # the quote itself, escaped
        pieces.append(Markup(f'<sup class="annotation-number">{number}</sup></mark>'))
        comments.append({"number": number, "label": label, "text": comment_text, "css_class": css_class})
        cursor = end
        number += 1
    pieces.append(escape(essay_text[cursor:]))  # whatever's left after the last quote

    return Markup("").join(pieces), comments

@app.route("/")  # runs when a browser visits the root URL ("/")
def home():
    return render_template("index.html", min_words=MIN_WORD_COUNT, max_words=MAX_WORD_COUNT)

@app.route("/topics")  # GET-only: just displays the topic-comparison form
def topics():
    return render_template("topics.html", min_topics=MIN_TOPICS, max_topics=MAX_TOPICS)

@app.route("/privacy")  # GET-only: static page, no form data involved
def privacy():
    return render_template("privacy.html")

@app.route("/analyze-topics", methods=["POST"])  # POST-only: receives the submitted topic list
def analyze_topics():
    raw_topics = request.form.getlist("topics")  # every field named "topics", regardless of how many exist
    topics_list = [t.strip() for t in raw_topics if t.strip()]  # drop empty/whitespace-only entries

    render_kwargs = {"min_topics": MIN_TOPICS, "max_topics": MAX_TOPICS}

    if not (MIN_TOPICS <= len(topics_list) <= MAX_TOPICS):
        return render_template(
            "topics.html",
            error=f"Please provide between {MIN_TOPICS} and {MAX_TOPICS} topics.",
            **render_kwargs,
        ), 400

    if _daily_limit_reached():
        return render_template(
            "topics.html",
            error="We've reached today's AI usage limit. Please check back tomorrow.",
            **render_kwargs,
        ), 503

    try:
        analysis = compare_topics(topics_list)  # calls Gemini and returns the parsed comparison dict
    except ValueError as e:  # raised when Gemini's response can't be parsed as JSON
        print(f"Topic analysis failed to parse: {e}")  # full detail logged server-side only
        return render_template(
            "topics.html", error="The AI response could not be understood. Please try again.", **render_kwargs
        ), 500

    return render_template("topics.html", analysis=analysis, **render_kwargs)

@app.route("/submit", methods=["POST"])  # only accepts POST requests (form submissions), not plain page visits
def submit():
    essay_text = request.form.get("essay", "")  # pulls the "essay" field from the submitted form; "" if missing
    return render_template("submitted.html", essay=essay_text)  # passes essay_text into the template as {{ essay }}

@app.route("/analyze", methods=["POST"])  # POST-only endpoint that grades the essay and shows the results page
def analyze():
    essay_text = request.form.get("essay", "").strip()  # .strip() removes leading/trailing whitespace
    essay_prompt = request.form.get("essay_prompt", "").strip()  # optional; "" if left blank

    # Bundles the two constants into every index.html render below, so the word-count
    # notice on the page always matches these values without repeating them elsewhere.
    render_kwargs = {"min_words": MIN_WORD_COUNT, "max_words": MAX_WORD_COUNT}

    if not essay_text:  # catches both "missing entirely" and "just spaces" cases
        return render_template("index.html", error="Essay text cannot be empty.", **render_kwargs), 400

    word_count = len(essay_text.split())  # .split() with no arguments splits on any whitespace

    if word_count < MIN_WORD_COUNT:
        return render_template(
            "index.html",
            error=f"Your essay is too short ({word_count} words). Please submit at least {MIN_WORD_COUNT} words.",
            **render_kwargs,
        ), 400

    if word_count > MAX_WORD_COUNT:
        return render_template(
            "index.html",
            error=f"Your essay is too long ({word_count} words). Please keep it under {MAX_WORD_COUNT} words.",
            **render_kwargs,
        ), 400

    if _daily_limit_reached():
        return render_template(
            "index.html",
            error="We've reached today's AI usage limit. Please check back tomorrow.",
            **render_kwargs,
        ), 503

    try:
        result = grade_essay(essay_text, essay_prompt=essay_prompt or None)  # "" becomes None when blank
    except ValueError as e:  # raised by grade_essay() when Gemini's response can't be parsed as JSON
        print(f"Grading failed to parse: {e}")  # full detail (including raw response) logged server-side only
        return render_template(
            "index.html", error="The AI response could not be understood. Please try again.", **render_kwargs
        ), 500

    dimension_data = result.get("dimensions", {})
    # prompt_fit only exists in dimension_data when a prompt was given — append it last if present,
    # rather than hardcoding it into DIMENSION_KEYS, which is shared with the no-prompt case too.
    dimension_order = DIMENSION_KEYS + (["prompt_fit"] if "prompt_fit" in dimension_data else [])
    dimensions = [(key, dimension_data[key]) for key in dimension_order if key in dimension_data]

    deeper_ideas = result.get("where_you_could_go_deeper", [])
    annotated_essay, essay_comments = annotate_essay(essay_text, dimensions, deeper_ideas)

    return render_template(
        "results.html",
        annotated_essay=annotated_essay,
        essay_comments=essay_comments,
        dimensions=dimensions,
        dimension_labels=DIMENSION_LABELS,
        overall_score=result.get("overall_score", 0),
        score_band_label=result.get("score_band_label", ""),
        what_i_would_remember=result.get("what_i_would_remember", ""),
        admissions_reader_concerns=result.get("admissions_reader_concerns", []),
        high_impact_revisions=result.get("high_impact_revisions", []),
        deeper_ideas=deeper_ideas,
        overall_summary=result.get("overall_summary", ""),
    )

if __name__ == "__main__":  # ensures this only runs when you execute `python app.py` directly
    port = int(os.environ.get("PORT", 5001))  # Render sets PORT itself; 5001 is only used for local dev
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"  # off unless explicitly enabled — Render won't set this
    app.run(host="0.0.0.0", port=port, debug=debug)  # 0.0.0.0 makes the app reachable from outside the container
