function debounce(fn, delay) {
    let timeout;
    return (...args) => {
        clearTimeout(timeout);
        timeout = setTimeout(() => fn(...args), delay);
    };
}

class AppState {
    constructor(blocks, submitButton) {
        this.blocks = [];
        this.blocksByID = {};
        this.logitScale = [0.0, 0.0];
        this.hoverWord = null;
        this.selectedWord = null;
        this.wordPopup = null;
        this.editingPopup = false;
        this.selectedTokens = {};
        this.customTexts = [];
        this.tokenMap = {};

        for (const block of blocks) {
            this.addBlock(block);
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

    addBlock(block) {
        block.onUpdateValid = () => {
            this.onUpdateValid();
        };
        this.blocks.push(block);
        this.blocksByID[block.id] = block;
    }

    onUpdateValid() {
        this.submitButton.disabled = !this.blocks.every(block => block.valid);
    }

    onSubmit() {
        let operations = [];
        for (const block of this.blocks) {
            operations.push(block.onSubmit());
            block.onClearResult();
        }
        fetch("/completion", {
            method: "POST",
            body: JSON.stringify({ operations }),
        }).then(response => response.json()).then(data => {
            let minLogit = Infinity;
            let maxLogit = -Infinity;
            for (const operation of data) {
                if (operation.result.logits) {
                    for (const tokenLogits of operation.result.logits) {
                        for (const logit of tokenLogits) {
                            minLogit = Math.min(minLogit, logit[1]);
                            maxLogit = Math.max(maxLogit, logit[1]);
                        }
                    }
                }
            }
            this.logitScale = [minLogit, maxLogit];
            for (const operation of data) {
                const block = this.blocksByID[operation.id];
                if (block.type === "branch") {
                    for (const op of operation.result) {
                        for (const childBlock of op) {
                            for (const token of Object.keys(childBlock.result.token_map)) {
                                this.tokenMap[token] = childBlock.result.token_map[token];
                            }
                        }
                    }
                } else {
                    for (const token of Object.keys(operation.result.token_map)) {
                        this.tokenMap[token] = operation.result.token_map[token];
                    }
                }
                block.onResult(
                    operation.result,
                    minLogit,
                    maxLogit,
                    (word, isHover) => isHover ? this.setHoverWord(word) : this.handleWordSelection(word, false));
            }
        });
    }

    renderPopup() {
        const word = this.selectedWord || this.hoverWord;
        this.wordPopup.innerHTML = word.getPopupHTML(this.selectedTokens, this.customTexts, this.logitScale[0], this.logitScale[1]);

        if (this.editingPopup) {
            this.wordPopup.classList.add("editing");
        } else {
            this.wordPopup.classList.remove("editing");
        }

        this.wordPopup.querySelector("button.add").addEventListener("click", () => {
            this.customTexts.push("");
            this.renderPopup();
        });

        this.wordPopup.querySelectorAll(".word-text input").forEach((input, idx) => {
            input.addEventListener("change", () => {
                this.selectedTokens[input.dataset.token] = input.checked;
            });
        });
        this.wordPopup.querySelectorAll(".word-input input").forEach((input, idx) => {
            input.addEventListener("change", () => {
                this.customTexts[idx] = input.value;
            });
        });
        this.wordPopup.querySelectorAll("button.delete").forEach((button, idx) => {
            button.addEventListener("click", () => {
                this.customTexts.splice(idx, 1);
                this.renderPopup();
            });
        });
        if (word.isEditable) {
            this.wordPopup.classList.add("editable");
            this.wordPopup.querySelector("button.edit").addEventListener("click", () => {
                this.editingPopup = true;
                this.renderPopup();
            });
        } else {
            this.wordPopup.classList.remove("editable");
        }
        this.wordPopup.querySelector("button.cancel").addEventListener("click", () => {
            this.editingPopup = false;
            this.selectedTokens = {};
            this.customTexts = [];
            this.renderPopup();
        });
        this.wordPopup.querySelector("button.apply").addEventListener("click", () => {
            let rawText = "";
            let tokenized = [];
            for (let i = 0; i < word.index; i++) {
                rawText += this.tokenMap[word.block.cachedLogits[i][0][0]];
                tokenized.push(word.block.cachedLogits[i][0][0]);
            }
            word.block.setType("text");
            word.block.updateText(rawText);
            word.block.updateTokenized(tokenized);
            word.block.expand();

            let options = [];
            for (const token of Object.keys(this.selectedTokens)) {
                options.push({
                    "raw": this.tokenMap[token],
                    "tokenized": [token],
                });
            }
            for (const customText of this.customTexts) {
                options.push({
                    "raw": customText,
                    "tokenized": null,
                });
            }

            // Generate a new block ID
            let newBlockID = "branch";
            while (this.blocksByID[newBlockID] !== undefined) {
                newBlockID = "branch-" + Math.random().toString(36).substring(2, 15);
            }
            for (let idx = 0; idx < this.blocks.length; idx++) {
                if (this.blocks[idx] === word.block) {
                    // Insert a new block after the current block
                    const blocks = document.getElementById("conversation-blocks");
                    const newBlock = new ConversationBlock(newBlockID, "assistant", "branch");
                    newBlock.updateBranchOptions(options);
                    blocks.insertBefore(newBlock, blocks.children[idx + 1]);
                    this.addBlock(newBlock);
                    break;
                }
            }

            this.editingPopup = false;
            this.selectedTokens = {};
            this.customTexts = [];
            this.renderPopup();
            this.onUpdateValid();
            //word.block.applyCustomTexts(word.index, this.selectedTokens, this.customTexts, text);
            //this.onSubmit();
        });
    }

    handleWordSelection(word, isHover) {
        if (isHover && this.selectedWord !== null) {
            return;
        }
        if (word === null) {
            this.hoverWord = null;
            this.selectedWord = null;
        } else if (isHover) {
            if (word == this.hoverWord) {
                return;
            }
            this.hoverWord = word;
        } else {
            if (word == this.selectedWord) {
                return;
            }
            this.selectedWord = word;
        }

        if (this.wordPopup !== null) {
            this.wordPopup.remove();
            this.wordPopup = null;
        }

        if (word !== null) {
            this.selectedTokens = {};
            this.customTexts = [];
            this.editingPopup = false;
            this.wordPopup = document.createElement("div");
            this.wordPopup.classList.add("word-popup");
            if (this.selectedWord === word) {
                this.wordPopup.classList.add("selected");
            }
            this.renderPopup();

            word.appendChild(this.wordPopup);
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

function getColor(logitValue, minLogit, maxLogit) {
    const normalizedLogitValue = (logitValue - maxLogit) * (COLORS.length - 1) / (minLogit - maxLogit);
    const colorIdx = Math.min(COLORS.length - 2, Math.floor(normalizedLogitValue));
    const colorFrac = normalizedLogitValue - colorIdx;
    const h = Math.round(COLORS[colorIdx][0] + colorFrac * (COLORS[colorIdx + 1][0] - COLORS[colorIdx][0]));
    const s = Math.round(COLORS[colorIdx][1] + colorFrac * (COLORS[colorIdx + 1][1] - COLORS[colorIdx][1]));
    const l = Math.round(COLORS[colorIdx][2] + colorFrac * (COLORS[colorIdx + 1][2] - COLORS[colorIdx][2]));
    return `hsl(${h}, ${s}%, ${l}%)`;
}

class Word extends HTMLDivElement {
    constructor(block, index, text, token, logitValue, minLogit, maxLogit, logits, tokenMap, isEditable, callback) {
        super();
        this.block = block;
        this.index = index;
        this.text = text;
        this.token = token;
        this.logitValue = logitValue;
        this.logits = logits;
        this.tokenMap = tokenMap;
        this.isEditable = isEditable;

        const color = getColor(logitValue, minLogit, maxLogit);
        this.classList.add("word");
        this.innerHTML = "<span>" + text.replaceAll(" ", "&nbsp;") + "</span>";
        this.style.borderBottom = `2px solid ${color}`;
        if (text.startsWith(" ")) {
            this.style.paddingLeft = "8px";
        }
        if (text.endsWith(" ")) {
            this.style.paddingRight = "8px";
        }
        this.addEventListener("mouseover", (evt) => {
            callback(this, true);
            evt.stopPropagation();
        });
        this.addEventListener("click", (evt) => {
            callback(this, false);
            evt.stopPropagation();
        });
    }

    getPopupHTML(selectedTokens, customTexts, minLogit, maxLogit) {
        const rows = [];
        const softmax = [];
        // TODO: Make configurable?
        const temperature = 10.0;
        for (const logit of this.logits) {
            softmax.push(Math.exp(logit[1] * 1.0 / temperature));
        }
        const maxSoftmax = softmax.reduce((a, b) => Math.max(a, b), 0);
        for (let i = 0; i < softmax.length; i++) {
            softmax[i] /= maxSoftmax;
        }
        for (let i = 0; i < this.logits.length; i++) {
            const width = Math.max(1, Math.round(softmax[i] * 80));
            rows.push(
                (this.logits[i][0] === this.token ? `<tr class="selected">` : `<tr>`) + 
                `<td class="word-bar"><div style="width: ${width}px; height: 100%; background-color: ${getColor(this.logits[i][1], minLogit, maxLogit)}"></td>` + 
                `<td class="word-text"><label><input type="checkbox" data-token="${this.logits[i][0]}" ${selectedTokens[this.logits[i][0]] ? "checked" : ""} />${this.tokenMap[this.logits[i][0]]}</label></td>` +
                `</tr>`);
        }
        rows.push(`<tr><td colspan="2" class="buttons"><button class="add">+ Add custom text</button></td></tr>`);
        for (let i = 0; i < customTexts.length; i++) {
            rows.push(`<tr><td colspan="2" class="word-input"><input id="input-${i}" value="${customTexts[i]}" /><button class="delete">🗑️</button></tr>`);
        }
        rows.push(
            `<tr><td colspan="2" class="buttons">` +
            `<button class="edit">Edit</button>` +
            `<button class="cancel">Cancel</button>` +
            `<button class="apply">Apply</button>` +
            `</td></tr>`);
        return "<table>" + rows.join("") + "</table>";
    }
}

const TYPE_LABELS = {
    "text": "Text",
    "completion": "Completion",
    "branch": "Branch",
};

class ConversationBlock extends HTMLDivElement {
    constructor(id, defaultRole, defaultType) {
        super();
        this.id = id;
        this.role = defaultRole;
        this.type = defaultType;
        this.parameters = {
            "text": {
                "raw": "",
                "tokenized": [],
            },
            "completion": {
                "maxTokens": 300,
            },
            "branch": {
                "options": [],
            },
        }
        this.cachedLogits = null;

        this.onUpdateValid = () => {};
        this.checkValid();

        this.classList.add("conversation-block");
        let templateContent = document.getElementById("conversation-block").content;
        this.appendChild(templateContent.cloneNode(true));

        this.setType(defaultType);
        this.setRole(defaultRole);

        const textInput = this.querySelector("textarea");
        textInput.addEventListener("input", debounce(() => {
            this.onTextInput(textInput.value);
        }, 300));

        this.querySelectorAll("input[name=role]").forEach(input => {
            input.addEventListener("change", () => {
                this.role = input.value;
                this.updateSummary();
            });
        });
        this.querySelectorAll("input[name=type]").forEach(input => {
            input.addEventListener("change", () => {
                this.type = input.value;
                this.updateSummary();
            });
        });
        this.querySelector("input[name=maxTokens]").addEventListener("change", () => {
            this.parameters["completion"]["maxTokens"] = parseInt(this.querySelector("input[name=maxTokens]").value);
            this.querySelector("div.parameter[data-type=completion] span:nth-child(2)").textContent = this.parameters["completion"]["maxTokens"];
            this.updateSummary();
        });

        this.updateSummary();
    }

    setType(type) {
        this.type = type;
        this.querySelector("input[name=type][value=" + type + "]").checked = true;
    }

    setRole(role) {
        this.role = role;
        this.querySelector("input[name=role][value=" + role + "]").checked = true;
    }

    onTextInput(value) {
        this.parameters["text"]["raw"] = value;

        // Send a request to the server to tokenize the input
        const indicator = this.querySelector(".indicator");
        indicator.classList.add("loading");
        indicator.textContent = "...";

        const formattedText = `<|${this.role}|>\n${value}<|end|>\n`;
        fetch("/tokenize", {
            method: "POST",
            body: JSON.stringify({ text: formattedText }),
        })
            .then(response => response.json())
            .then(data => {
                this.updateTokenized(data);
            });
    }

    updateText(text) {
        this.parameters["text"]["raw"] = text;
        this.querySelector("textarea").value = text;
    }

    updateTokenized(tokenized) {
        this.parameters["text"]["tokenized"] = tokenized;
        this.checkValid();
        const indicator = this.querySelector(".indicator");
        indicator.classList.remove("loading");
        indicator.classList.add("valid");
        indicator.innerHTML = `&#10004; ${tokenized.length} tokens`;
    }

    updateBranchOptions(options) {
        // TODO: Show tokenized indicator
        this.parameters["branch"]["options"] = options;
        let html = "";
        for (let idx = 0; idx < options.length; idx++) {
            html += `<label>Option ${idx+1}: <input name="branch-${idx}" value="${options[idx]["raw"]}" /></label>`;
        }
        this.querySelector("div.type-parameters[data-type=branch]").innerHTML = html;
        this.checkValid();
    }

    updateSummary() {
        this.querySelector("summary h2").textContent = `${TYPE_LABELS[this.type]} (${this.role})`;
    }

    checkValid() {
        if (this.type === "text") {
            this.valid = this.parameters["text"]["tokenized"].length > 0;
        } else if (this.type === "completion") {
            this.valid = true;
        } else if (this.type === "branch") {
            this.valid = this.parameters["branch"]["options"].length > 0;
        }
        this.onUpdateValid();
    }

    expand() {
        const detailsElement = this.querySelector("details");
        detailsElement.open = true;
    }

    onSubmit() {
        if (!this.valid) {
            return null;
        }
        const detailsElement = this.querySelector("details");
        detailsElement.open = false;
        if (this.type === "text") {
            return {
                id: this.id,
                name: "feed_tokens",
                feed_tokens: {
                    tokens: this.parameters["text"]["tokenized"],
                    top_p: 10,
                },
            };
        } else if (this.type === "completion") {
            return {
                id: this.id,
                name: "completion",
                completion: {
                    max_tokens: this.parameters["completion"]["maxTokens"],
                    top_p: 10,
                },
            };
        } else if (this.type === "branch") {
            let forks = [];
            for (let idx = 0; idx < this.parameters["branch"]["options"].length; idx++) {
                forks.push([
                    {
                        id: this.id + "-" + idx,
                        name: "feed_tokens",
                        feed_tokens: {
                            tokens: this.parameters["branch"]["options"][idx]["tokenized"],
                            top_p: 10,
                        },
                    },
                ]);
            }
            return {
                id: this.id,
                name: "branch",
                branch: {
                    forks,
                },
            };
        }
    }

    onClearResult() {
        const resultElement = this.querySelector(".result");
        resultElement.innerHTML = "";
        this.cachedLogits = null;
    }

    onResult(result, minLogit, maxLogit, callback) {
        this.cachedLogits = result.logits;

        const resultElement = this.querySelector(".result");
        resultElement.innerHTML = "";

        if (this.type === "branch") {
            for (let forkIdx = 0; forkIdx < result.length; forkIdx++) {
                for (const childBlock of result[forkIdx]) {
                    const div = document.createElement("div");
                    for (let tokenIdx = 0; tokenIdx < childBlock.result.logits.length; tokenIdx++) {
                        const text = childBlock.result.token_map[childBlock.result.logits[tokenIdx][0][0]];
                        const word = new Word(this, tokenIdx, text, childBlock.result.logits[tokenIdx][0][0], childBlock.result.logits[tokenIdx][0][1], minLogit, maxLogit, childBlock.result.logits[tokenIdx], childBlock.result.token_map, false, callback);
                        div.appendChild(word);
                    }
                    resultElement.appendChild(div);
                }
            }
            return;
        }

        for (let tokenIdx = 0; tokenIdx < result.logits.length; tokenIdx++) {
            if (this.type === "text") {
                let logitValue = 0.0;
                result.logits[tokenIdx].forEach(logit => {
                    if (logit[0] === this.parameters["text"]["tokenized"][tokenIdx]) {
                        logitValue = logit[1];
                    }
                });
                const text = result.token_map[this.parameters["text"]["tokenized"][tokenIdx]];
                const word = new Word(this, tokenIdx, text, this.parameters["text"]["tokenized"][tokenIdx], logitValue, minLogit, maxLogit, result.logits[tokenIdx], result.token_map, false, callback);
                resultElement.appendChild(word);
            } else if (this.type === "completion") {
                const text = result.token_map[result.logits[tokenIdx][0][0]];
                const word = new Word(this, tokenIdx, text, result.logits[tokenIdx][0][0], result.logits[tokenIdx][0][1], minLogit, maxLogit, result.logits[tokenIdx], result.token_map, true, callback);
                resultElement.appendChild(word);
            } 
        }
    }
}

document.addEventListener("DOMContentLoaded", () => {
    customElements.define("conversation-block", ConversationBlock, { extends: "div" });
    customElements.define("conversation-word", Word, { extends: "div" });

    const blocks = document.getElementById("conversation-blocks");

    const systemPrompt = new ConversationBlock("system-prompt", "system", "text");
    blocks.appendChild(systemPrompt);

    const userPrompt = new ConversationBlock("user-prompt", "user", "text");
    blocks.appendChild(userPrompt);

    const assistantResponse = new ConversationBlock("assistant-response", "assistant", "completion");
    blocks.appendChild(assistantResponse);

    const submitButton = document.getElementById("submit");
    const appState = new AppState([systemPrompt, userPrompt, assistantResponse], submitButton);
});
