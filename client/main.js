function debounce(fn, delay) {
    let timeout;
    return () => {
        clearTimeout(timeout);
        timeout = setTimeout(fn, delay);
    };
}

class AppState {
    constructor(prompts, submitButton) {
        this.prompts = [];
        this.promptsByID = {};
        for (const prompt of prompts) {
            this.addPrompt(prompt);
        }
        this.submitButton = submitButton;
        this.submitButton.addEventListener("click", () => {
            this.onSubmit();
        });
        this.onUpdateValid();
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
            for (const operation of data) {
                const prompt = this.promptsByID[operation.id];
                prompt.onResult(operation.result, min_logit, max_logit);
            }
        });
    }
}

const COLORS = [
    [246.7, 50.3, 100.0],
    [257.9, 65.8, 94.7],
    [212.0, 49.0, 82.6],
    [159.3, 30.2, 87.2],
    [37.8, 59.6, 94.1],
]

function getColor(logit_value, min_logit, max_logit) {
    const normalized_logit_value = (logit_value - min_logit) * (COLORS.length - 1) / (max_logit - min_logit);
    const color_idx = Math.min(COLORS.length - 2, Math.floor(normalized_logit_value));
    const color_frac = normalized_logit_value - color_idx;
    const h = COLORS[color_idx][0] + color_frac * (COLORS[color_idx + 1][0] - COLORS[color_idx][0]);
    const s = COLORS[color_idx][1] + color_frac * (COLORS[color_idx + 1][1] - COLORS[color_idx][1]);
    const l = COLORS[color_idx][2] + color_frac * (COLORS[color_idx + 1][2] - COLORS[color_idx][2]);
    return `hsl(${h}, ${s}%, ${l}%)`;
}

class Word {
    constructor(text, logit_value, min_logit, max_logit) {
        this.text = text;
        this.logit_value = logit_value;
        this.min_logit = min_logit;
        this.max_logit = max_logit;

        const color = getColor(logit_value, min_logit, max_logit);
        this.element = document.createElement("span");
        this.element.textContent = text;
        this.element.style.backgroundColor = color;
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
                this.indicator.innerHTML = "&#10004;";
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

    onResult(result, min_logit, max_logit) {
        this.inputElement.classList.add("hidden");
        this.resultElement.classList.remove("hidden");
        this.resultElement.innerHTML = "";
        for (let token_idx = 0; token_idx < result.logits.length; token_idx++) {
            let logit_value = 0.0;
            result.logits[token_idx].forEach(logit => {
                if (logit[0] === this.tokens[token_idx]) {
                    logit_value = logit[1];
                }
            });
            const text = result.token_map[this.tokens[token_idx]];
            const word = new Word(text, logit_value, min_logit, max_logit);
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

    onResult(result, min_logit, max_logit) {
        this.resultElement.classList.remove("hidden");
        this.resultElement.innerHTML = "";
        for (let token_idx = 0; token_idx < result.logits.length; token_idx++) {
            const text = result.token_map[result.logits[token_idx][0][0]];
            const word = new Word(text, result.logits[token_idx][0][1], min_logit, max_logit);
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
