// OperationEditor represents the editable version of an operation that is sent
// to the completion endpoint as part of a script. The script is a sequence of
// operations that can be nested inside each other.

const TYPE_LABELS = {
    "text": "Text",
    "completion": "Completion",
    "branch": "Branch",
};

class OperationEditor extends HTMLDivElement {
    constructor(id, type, defaultRole, templateId, defaultParameters) {
        super();
        this.type = type;
        this.role = defaultRole;
        this.parameters = defaultParameters;

        this.classList.add("operation-editor");
        let templateContent = document.getElementById(templateId).content;
        this.appendChild(templateContent.cloneNode(true));

        this.onUpdate = () => {};
        this.checkValid();

        this.setId(id);
        this.setRole(defaultRole);

        this.querySelectorAll("input[name=role]").forEach(input => {
            input.addEventListener("change", () => {
                this.role = input.value;
            this.checkValid();
            });
        });
    }

    setId(id) {
        this.id = id;
        this.querySelector("label.index").textContent = '#' + id;
    }

    setRole(role) {
        this.role = role;
        this.querySelector("input[name=role][value=" + role + "]").checked = true;
    }
}

class OperationEditorText extends OperationEditor {
    constructor(id, defaultRole, params) {
        super(id, "text", defaultRole, "operation-editor-text", params || {
            "raw": "",
            "tokenized": [],
        });

        const textInput = this.querySelector("textarea");
        textInput.value = this.parameters["raw"];
        textInput.addEventListener("input", debounce(() => {
            this.onTextInput(textInput.value);
        }, 300));
        this.updateTokenized(this.parameters["tokenized"]);
    }

    checkValid() {
        this.valid = this.parameters["tokenized"].length > 0;
        this.onUpdate();
    }


    onTextInput(value) {
        this.parameters["raw"] = value;

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
        this.parameters["raw"] = text;
        this.querySelector("textarea").value = text;
        this.checkValid();
    }

    updateTokenized(tokenized) {
        this.parameters["tokenized"] = tokenized;
        this.checkValid();
        const indicator = this.querySelector(".indicator");
        indicator.classList.remove("loading");
        if (this.valid) {
            indicator.classList.add("valid");
        } else {
            indicator.classList.remove("valid");
        }
        indicator.innerHTML = `&#10004; ${tokenized.length} tokens`;
    }

    onSubmit() {
        if (!this.valid) {
            return null;
        }
        return {
            id: `${this.id}`,
            name: "feed_tokens",
            feed_tokens: {
                tokens: this.parameters["tokenized"],
                top_p: 10,
            },
        };
    }
}

class OperationEditorCompletion extends OperationEditor {
    constructor(id, defaultRole, params) {
        super(id, "completion", defaultRole, "operation-editor-completion", params || {
            "maxTokens": 300,
        });

        const maxTokens = this.querySelector("input[name=maxTokens]");
        maxTokens.value = this.parameters["maxTokens"];
        maxTokens.addEventListener("change", () => {
            this.parameters["maxTokens"] = parseInt(maxTokens.value);
            this.checkValid();
        });
    }

    checkValid() {
        this.valid = true;
        this.onUpdate();
    }

    onSubmit() {
        if (!this.valid) {
            return null;
        }
        return {
            id: `${this.id}`,
            name: "completion",
            completion: {
                max_tokens: this.parameters["maxTokens"],
                top_p: 10,
            },
        };
    }
}

class OperationEditorBranch extends OperationEditor {
    constructor(id, defaultRole, params) {
        super(id, "branch", defaultRole, "operation-editor-branch", params || {
            "options": [],
        });
    }

    checkValid() {
        this.valid = this.parameters["branch"]["options"].length > 0;
        this.onUpdate();
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

    onSubmit() {
        if (!this.valid) {
            return null;
        }
        let forks = [];
        for (let idx = 0; idx < this.parameters["options"].length; idx++) {
            forks.push([
                {
                    id: `${this.id}:${idx}`,
                    name: "feed_tokens",
                    feed_tokens: {
                        tokens: this.parameters["options"][idx]["tokenized"],
                        top_p: 10,
                    },
                },
            ]);
        }
        return {
            id: `${this.id}`,
            name: "branch",
            branch: {
                forks,
            },
        };
    }
}

customElements.define("operation-editor", OperationEditor, { extends: "div" });
customElements.define("operation-editor-text", OperationEditorText, { extends: "div" });
customElements.define("operation-editor-completion", OperationEditorCompletion, { extends: "div" });
customElements.define("operation-editor-branch", OperationEditorBranch, { extends: "div" });
