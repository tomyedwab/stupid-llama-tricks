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
    constructor(index, text, token, logitValue, minLogit, maxLogit, logits, tokenMap, isEditable, callback) {
        super();
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

customElements.define("conversation-word", Word, { extends: "div" });

class Results extends HTMLDivElement {
    constructor(responseData) {
        super();

        this.tokenMap = {};
        this.logitScale = [0.0, 0.0];

        this.hoverWord = null;
        this.selectedWord = null;
        this.wordPopup = null;
        this.editingPopup = false;
        this.selectedTokens = {};
        this.customTexts = [];

        this.setHoverWord = debounce((word) => {
            this.handleWordSelection(word, true);
        }, 300);

        let minLogit = Infinity;
        let maxLogit = -Infinity;
        for (const operation of responseData) {
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

        const callback = (word, isHover) => isHover ? this.setHoverWord(word) : this.handleWordSelection(word, false);

        for (const operation of responseData) {
            if (operation.name === "branch") {
                for (const fork of operation.result) {
                    for (const childOperation of fork) {
                        for (const token of Object.keys(childOperation.result.token_map)) {
                            this.tokenMap[token] = childOperation.result.token_map[token];
                        }

                        const div = document.createElement("div");
                        for (let tokenIdx = 0; tokenIdx < childOperation.result.logits.length; tokenIdx++) {
                            const text = childOperation.result.token_map[childOperation.result.logits[tokenIdx][0][0]];
                            const word = new Word(tokenIdx, text, childOperation.result.logits[tokenIdx][0][0], childOperation.result.logits[tokenIdx][0][1], minLogit, maxLogit, childOperation.result.logits[tokenIdx], childOperation.result.token_map, false, callback);
                            div.appendChild(word);
                        }
                        this.appendChild(div);
                    }
                }

            } else {
                const blockLabel = document.createElement("div");
                blockLabel.classList.add("block-label");
                blockLabel.textContent = '#' + operation.id;
                this.appendChild(blockLabel);

                for (const token of Object.keys(operation.result.token_map)) {
                    this.tokenMap[token] = operation.result.token_map[token];
                }
                for (let tokenIdx = 0; tokenIdx < operation.result.logits.length; tokenIdx++) {
                    if (operation.name === "feed_tokens") {
                        let logitValue = 0.0;
                        operation.result.logits[tokenIdx].forEach(logit => {
                            if (logit[0] === operation.feed_tokens.tokens[tokenIdx]) {
                                logitValue = logit[1];
                            }
                        });
                        const text = operation.result.token_map[operation.feed_tokens.tokens[tokenIdx]];
                        const word = new Word(tokenIdx, text, operation.feed_tokens.tokens[tokenIdx], logitValue, minLogit, maxLogit, operation.result.logits[tokenIdx], operation.result.token_map, false, callback);
                        if (text.startsWith("\n")) {
                            this.appendChild(document.createElement("br"));
                        }
                        this.appendChild(word);
                        if (text.endsWith("\n") || text.endsWith("<|end|>")) {
                            this.appendChild(document.createElement("br"));
                        }
                    } else if (operation.name === "completion") {
                        const text = operation.result.token_map[operation.result.logits[tokenIdx][0][0]];
                        const word = new Word(tokenIdx, text, operation.result.logits[tokenIdx][0][0], operation.result.logits[tokenIdx][0][1], minLogit, maxLogit, operation.result.logits[tokenIdx], operation.result.token_map, true, callback);
                        if (text.startsWith("\n")) {
                            this.appendChild(document.createElement("br"));
                        }
                        this.appendChild(word);
                        if (text.endsWith("\n") || text.endsWith("<|end|>")) {
                            this.appendChild(document.createElement("br"));
                        }
                    } 
                }
            }

        }
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
            /* TODO(tom)
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
            */
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
