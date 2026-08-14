// This file is loaded via a <script> tag placed just before </body> in index.html,
// so by the time this code runs, the form/button/spinner elements below already exist
// in the page — no need to wait for a "page loaded" event first.

const form = document.querySelector("form");                 // the essay submission form
const button = form.querySelector("button[type='submit']");  // the submit button inside it
const spinner = document.getElementById("spinner");          // the loading spinner element

form.addEventListener("submit", () => {
    button.disabled = true;             // prevents a second click while the request is in flight
    button.textContent = "Grading...";  // gives clear feedback that something is happening
    spinner.hidden = false;             // reveals the spinner (the form still submits normally after this)
});
