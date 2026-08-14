// Loaded via a <script> tag at the end of <body> in topics.html, so these elements
// already exist in the page by the time this code runs.

const container = document.getElementById("topics-container");  // holds all the topic textareas
const addButton = document.getElementById("add-topic-btn");      // the "+ Add another topic" button

const MAX_TOPICS = 5;  // matches MAX_TOPICS in app.py; kept in sync manually since JS and Python can't share a constant

addButton.addEventListener("click", () => {
    const currentCount = container.querySelectorAll("textarea").length;  // how many topic boxes exist right now

    if (currentCount >= MAX_TOPICS) {
        return;  // safety guard; the button should already be disabled by the time this could happen
    }

    const newTextarea = document.createElement("textarea");  // builds a new, empty <textarea> element
    newTextarea.name = "topics";  // same name as the existing boxes, so Flask reads it as part of the same list
    newTextarea.rows = 2;
    newTextarea.placeholder = "Describe a potential essay topic...";
    newTextarea.required = true;

    container.appendChild(newTextarea);  // inserts the new textarea into the page

    if (container.querySelectorAll("textarea").length >= MAX_TOPICS) {
        addButton.disabled = true;  // once at the max, prevent adding any more
    }
});
