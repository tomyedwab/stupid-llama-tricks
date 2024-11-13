const COLORS = [
    [246, 16, 24],
    [258, 83, 50],
    [212, 55, 60],
    [92, 92, 70],
    [5, 87, 94],
]

function getColor(logitValue, logitScale) {
    const normalizedLogitValue = (logitValue - logitScale[1]) * (COLORS.length - 1) / (logitScale[0] - logitScale[1]);
    const colorIdx = Math.min(COLORS.length - 2, Math.floor(normalizedLogitValue));
    const colorFrac = normalizedLogitValue - colorIdx;
    const h = Math.round(COLORS[colorIdx][0] + colorFrac * (COLORS[colorIdx + 1][0] - COLORS[colorIdx][0]));
    const s = Math.round(COLORS[colorIdx][1] + colorFrac * (COLORS[colorIdx + 1][1] - COLORS[colorIdx][1]));
    const l = Math.round(COLORS[colorIdx][2] + colorFrac * (COLORS[colorIdx + 1][2] - COLORS[colorIdx][2]));
    return `hsl(${h}, ${s}%, ${l}%)`;
}

class Word extends HTMLDivElement {
    constructor(operationId, index, token, logits, tokenMap, isEditable, callback) {
        super();
        this.operationId = operationId;
        this.index = index;
        this.text = tokenMap[token];
        this.token = token;
        this.logits = logits;
        this.tokenMap = tokenMap;
        this.isEditable = isEditable;

        this.tokenLogit = 0.0;
        for (const logit of logits) {
            if (logit[0] === token) {
                this.tokenLogit = logit[1];
            }
        }

        this.classList.add("word");

        if (this.text === "<|end|>") {
            return;
        } else if (this.text === "<|system|>") {
            this.classList.add("role-label");
            this.innerHTML = "System:";
        } else if (this.text === "<|user|>") {
            this.classList.add("role-label");
            this.innerHTML = "User:";
        } else if (this.text === "<|assistant|>") {
            this.classList.add("role-label");
            this.innerHTML = "Assistant:";
        } else {
            this.innerHTML = "<span>" + this.text.replaceAll(" ", "&nbsp;") + "</span>";
        }

        if (this.text.startsWith(" ")) {
            this.style.paddingLeft = "8px";
        }
        if (this.text.endsWith(" ")) {
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

    updateColor(logitScale) {
        const color = getColor(this.tokenLogit, logitScale);
        this.style.borderBottom = `2px solid ${color}`;
    }

    getPopupHTML(selectedTokens, customTexts, logitScale) {
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
                `<td class="word-bar"><div style="width: ${width}px; height: 100%; background-color: ${getColor(this.logits[i][1], logitScale)}"></td>` + 
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

customElements.define("conversation-word", Word, { extends: "div" });

class Results extends HTMLDivElement {
    constructor(operations, tokenMap) {
        super();

        this.logitScale = [0.0, 0.0];

        this.hoverWord = null;
        this.selectedWord = null;
        this.wordPopup = null;
        this.editingPopup = false;
        this.selectedTokens = {};
        this.customTexts = [];
        this.cachedLogits = {};
        this.operationDivs = {};
        this.tokenMap = tokenMap;

        this.onApplyEdits = () => {};

        this.setHoverWord = debounce((word) => {
            this.handleWordSelection(word, true);
        }, 300);

        this.logitScale = [Infinity, -Infinity];

        this.callback = (word, isHover) => isHover ? this.setHoverWord(word) : this.handleWordSelection(word, false);

        for (const operation of operations) {
            this.appendChild(this.createOperationDiv(operation));
        }
    }

    createOperationDiv(operation) {
        const div = document.createElement("div");
        div.classList.add("operation");

        const blockLabel = document.createElement("div");
        blockLabel.classList.add("block-label");
        blockLabel.textContent = '#' + operation.id;
        div.appendChild(blockLabel);

        if (operation instanceof OperationEditorBranch) {
            for (const option of operation.options) {
                for (const childOperation of option.operations) {
                    const childDiv = this.createOperationDiv(childOperation);
                    div.appendChild(childDiv);
                }
            }
        }

        this.operationDivs[operation.id] = {
            div,
            nextIndex: 0,
            tokenQueue: [],
            words: [],
        };

        return div;
    }

    addToken(operationId, tokenIndex, tokenId, logits) {
        const operationDiv = this.operationDivs[operationId];
        operationDiv.tokenQueue.push([
            tokenIndex,
            tokenId,
            logits,
        ]);
        let updatedLogits = false;
        for (const logit of logits) {
            if (logit[1] < this.logitScale[0]) {
                this.logitScale[0] = logit[1];
                updatedLogits = true;
            }
            if (logit[1] > this.logitScale[1]) {
                this.logitScale[1] = logit[1];
                updatedLogits = true;
            }
        }
        if (operationDiv.tokenQueue.length > 1) {
            operationDiv.tokenQueue.sort((a, b) => a[0] - b[0]);
        }
        while (operationDiv.tokenQueue.length > 0 && operationDiv.tokenQueue[0][0] === operationDiv.nextIndex) {
            const token = operationDiv.tokenQueue.shift();
            const word = this.renderToken(operationId, token[0], token[1], token[2]);
            if (!updatedLogits && this.logitScale[1] > this.logitScale[0]) {
                word.updateColor(this.logitScale);
            }
            operationDiv.nextIndex++;
        }
        if (updatedLogits && this.logitScale[1] > this.logitScale[0]) {
            for (const otherDiv of Object.values(this.operationDivs)) {
                for (const word of otherDiv.words) {
                    word.updateColor(this.logitScale);
                }
            }
        }
    }

    renderToken(operationId, tokenIndex, tokenId, logits) {
        const word = new Word(operationId, tokenIndex, tokenId, logits, this.tokenMap, false, this.callback);
        this.operationDivs[operationId].words.push(word);
        this.operationDivs[operationId].div.appendChild(word);
        return word;
    }

    renderPopup() {
        const word = this.selectedWord || this.hoverWord;
        this.wordPopup.innerHTML = word.getPopupHTML(this.selectedTokens, this.customTexts, this.logitScale);

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
                rawText += this.tokenMap[this.cachedLogits[word.operationId][i][0][0]];
                tokenized.push(this.cachedLogits[word.operationId][i][0][0]);
            }

            const branchOptions = [];
            for (const token of Object.keys(this.selectedTokens)) {
                branchOptions.push({"operations": [["text", "assistant", {
                    "raw": this.tokenMap[token],
                    "tokenized": [token],
                }]]});
            }
            for (const customText of this.customTexts) {
                branchOptions.push({"operations": [["text", "assistant", {
                    "raw": customText,
                    "tokenized": [],
                }]]});
            }

            this.onApplyEdits(word.operationId, {
                "raw": rawText,
                "tokenized": tokenized,
            }, {
                "options": branchOptions,
            });

            this.editingPopup = false;
            this.selectedTokens = {};
            this.customTexts = [];
            this.renderPopup();
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
};

customElements.define("conversation-results", Results, { extends: "div" });
