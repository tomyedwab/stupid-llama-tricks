function debounce(fn, delay) {
    let timeout;
    return (...args) => {
        clearTimeout(timeout);
        timeout = setTimeout(() => fn(...args), delay);
    };
}

class AppState {
    constructor(prompts, submitButton) {
        this.prompts = [];
        this.promptsByID = {};
        this.logitScale = [0.0, 0.0];
        this.hoverWord = null;
        this.selectedWord = null;
        this.wordPopup = null;
        for (const prompt of prompts) {
            this.addPrompt(prompt);
        }
        this.submitButton = submitButton;
        this.submitButton.addEventListener("click", () => {
            this.onSubmit();
        });
        this.onUpdateValid();

        this.setHoverWord = debounce((word) => {
            this.handleWordSelection(word, true);
        }, 300);

        document.body.addEventListener("click", () => {
            this.handleWordSelection(null, false);
        });
    }

    addPrompt(prompt) {
        prompt.onUpdateValid = () => {
            this.onUpdateValid();
        };
        this.prompts.push(prompt);
        this.promptsByID[prompt.id] = prompt;
    }

    onUpdateValid() {
        this.submitButton.disabled = !this.prompts.every(prompt => prompt.valid);
    }

    onSubmit() {
        let operations = [];
        for (const prompt of this.prompts) {
            operations.push(prompt.onSubmit());
        }
        fetch("/completion", {
            method: "POST",
            body: JSON.stringify({ operations }),
        }).then(response => response.json()).then(data => {
            let min_logit = Infinity;
            let max_logit = -Infinity;
            for (const operation of data) {
                if (operation.result.logits) {
                    for (const token_logits of operation.result.logits) {
                        for (const logit of token_logits) {
                            min_logit = Math.min(min_logit, logit[1]);
                            max_logit = Math.max(max_logit, logit[1]);
                        }
                    }
                }
            }
            this.logitScale = [min_logit, max_logit];
            for (const operation of data) {
                const prompt = this.promptsByID[operation.id];
                prompt.onResult(
                    operation.result,
                    min_logit,
                    max_logit,
                    (word, is_hover) => is_hover ? this.setHoverWord(word) : this.handleWordSelection(word, false));
            }
        });
    }

    handleWordSelection(word, is_hover) {
        if (is_hover && this.selectedWord !== null) {
            return;
        }
        if (word === null) {
            this.hoverWord = null;
            this.selectedWord = null;
        } else if (is_hover) {
            this.hoverWord = word;
        } else {
            this.selectedWord = word;
        }

        if (this.wordPopup !== null) {
            this.wordPopup.remove();
            this.wordPopup = null;
        }

        if (word !== null) {
            this.wordPopup = document.createElement("div");
            this.wordPopup.classList.add("word-popup");
            this.wordPopup.innerHTML = word.getPopupHTML(this.logitScale[0], this.logitScale[1]);
            word.element.appendChild(this.wordPopup);
        }
    }
}

const COLORS = [
    [246, 16, 24],
    [258, 83, 50],
    [212, 55, 60],
    [92, 92, 70],
    [5, 87, 94],
]

function getColor(logit_value, min_logit, max_logit) {
    const normalized_logit_value = (logit_value - max_logit) * (COLORS.length - 1) / (min_logit - max_logit);
    const color_idx = Math.min(COLORS.length - 2, Math.floor(normalized_logit_value));
    const color_frac = normalized_logit_value - color_idx;
    const h = Math.round(COLORS[color_idx][0] + color_frac * (COLORS[color_idx + 1][0] - COLORS[color_idx][0]));
    const s = Math.round(COLORS[color_idx][1] + color_frac * (COLORS[color_idx + 1][1] - COLORS[color_idx][1]));
    const l = Math.round(COLORS[color_idx][2] + color_frac * (COLORS[color_idx + 1][2] - COLORS[color_idx][2]));
    return `hsl(${h}, ${s}%, ${l}%)`;
}

class Word {
    constructor(text, logit_value, min_logit, max_logit, logits, token_map, callback) {
        this.text = text;
        this.logit_value = logit_value;
        this.logits = logits;
        this.token_map = token_map;

        const color = getColor(logit_value, min_logit, max_logit);
        this.element = document.createElement("div");
        this.element.classList.add("word");
        this.element.textContent = text;
        this.element.style.borderBottom = `2px solid ${color}`;
        if (text.startsWith(" ")) {
            this.element.style.marginLeft = "8px";
        }
        if (text.endsWith(" ")) {
            this.element.style.marginRight = "8px";
        }
        this.element.addEventListener("mouseover", (evt) => {
            callback(this, true);
            evt.stopPropagation();
        });
        this.element.addEventListener("click", (evt) => {
            callback(this, false);
            evt.stopPropagation();
        });
    }

    getPopupHTML(min_logit, max_logit) {
        const rows = [];
        const softmax = [];
        // TODO: Make configurable?
        const temperature = 10.0;
        for (const logit of this.logits) {
            softmax.push(Math.exp(logit[1] * 1.0 / temperature));
        }
        const max_softmax = softmax.reduce((a, b) => Math.max(a, b), 0);
        for (let i = 0; i < softmax.length; i++) {
            softmax[i] /= max_softmax;
        }
        for (let i = 0; i < this.logits.length; i++) {
            const width = Math.max(1, Math.round(softmax[i] * 80));
            rows.push(
                `<tr>` + 
                `<td class="word-bar"><div style="width: ${width}px; height: 100%; background-color: ${getColor(this.logits[i][1], min_logit, max_logit)}"></td>` + 
                `<td class="word-text">${this.token_map[this.logits[i][0]]}</td>` +
                `</tr>`);
        }
        return "<table>" + rows.join("") + "</table>";
    }
}

class TextInput {
    constructor(id, element, role) {
        this.id = id;
        this.inputElement = element.querySelector("textarea");
        this.resultElement = element.querySelector(".result");
        this.indicator = element.querySelector(".indicator");
        this.role = role;
        this.valid = false;
        this.tokens = [];
        this.onUpdateValid = () => {};
        this.inputElement.addEventListener("input", debounce(() => {
            this.onInput(this.inputElement.value);
        }, 300));
        if (this.inputElement.value !== "") {
            this.onInput(this.inputElement.value);
        }
    }

    onInput(value) {
        // Send a request to the server to tokenize the input
        this.indicator.classList.add("loading");
        this.indicator.textContent = "...";

        const formattedText = `<|${this.role}|>\n${value}<|end|>\n`;
        fetch("/tokenize", {
            method: "POST",
            body: JSON.stringify({ text: formattedText }),
        })
            .then(response => response.json())
            .then(data => {
                this.tokens = data;
                this.valid = data.length > 0;
                this.onUpdateValid();
                this.indicator.classList.remove("loading");
                this.indicator.classList.add("valid");
                this.indicator.innerHTML = `&#10004; ${data.length} tokens`;
            });
    }

    onSubmit() {
        this.inputElement.disabled = true;
        return {
            id: this.id,
            name: "feed_tokens",
            feed_tokens: {
                tokens: this.tokens,
                top_p: 10,
            },
        }
    }

    onResult(result, min_logit, max_logit, callback) {
        this.resultElement.innerHTML = "";
        for (let token_idx = 0; token_idx < result.logits.length; token_idx++) {
            let logit_value = 0.0;
            result.logits[token_idx].forEach(logit => {
                if (logit[0] === this.tokens[token_idx]) {
                    logit_value = logit[1];
                }
            });
            const text = result.token_map[this.tokens[token_idx]];
            const word = new Word(text, logit_value, min_logit, max_logit, result.logits[token_idx], result.token_map, callback);
            this.resultElement.appendChild(word.element);
        }
    }
}

class AssistantResponse {
    constructor(id, element) {
        this.id = id;
        this.onUpdateValid = () => {};
        this.valid = true;
        this.resultElement = element.querySelector(".result");
    }

    onSubmit() {
        return {
            id: this.id,
            name: "completion",
            completion: {
                // TODO: Add configuration option
                max_tokens: 1000,
                top_p: 10,
            },
        }
    }

    onResult(result, min_logit, max_logit, callback) {
        this.resultElement.innerHTML = "";
        for (let token_idx = 0; token_idx < result.logits.length; token_idx++) {
            const text = result.token_map[result.logits[token_idx][0][0]];
            const word = new Word(text, result.logits[token_idx][0][1], min_logit, max_logit, result.logits[token_idx], result.token_map, callback);
            this.resultElement.appendChild(word.element);
        }
    }
}

document.addEventListener("DOMContentLoaded", () => {
    const systemPrompt = new TextInput("system-prompt", document.getElementById("system-prompt"), "system");
    const userPrompt = new TextInput("user-prompt", document.getElementById("user-prompt"), "user");
    const assistantResponse = new AssistantResponse("assistant-response", document.getElementById("assistant-response"));
    const submitButton = document.getElementById("submit");
    const appState = new AppState([systemPrompt, userPrompt, assistantResponse], submitButton);
});
