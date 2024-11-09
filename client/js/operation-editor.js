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
        this.onAdd = () => {};
        this.onRemove = () => {};

        this.checkValid();

        this.setId(id);
        this.setRole(defaultRole);

        this.querySelectorAll("input[name=role]").forEach(input => {
            input.addEventListener("change", () => {
                this.role = input.value;
            this.checkValid();
            });
        });

        this.querySelector(".add").addEventListener("click", () => {
            this.onAdd(this.id, null);
        });
        this.querySelector(".remove").addEventListener("click", () => {
            this.onRemove(this.id);
        });
        this.querySelector(".help").addEventListener("click", () => {
            // TODO: Show help text somewhere
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

    forEachChild(callback) {
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

class OperationEditorChildOperations extends HTMLDivElement {
    constructor(id, parameters, onUpdate, onAdd, onRemove) {
        super();
        this.id = id;
        this.operations = [];
        this.parameters = {
            "operations": [],
        };
        this.onUpdate = onUpdate;
        this.onAdd = onAdd;

        this.classList.add("child-operations");
        this.classList.add("visible-branch");
        this.setAttribute("data-id", id);

        if (parameters.operations.length > 0) {
            for (const [type, role, child_parameters] of parameters.operations) {
                if (type === "text") {
                    this.addOperation(new OperationEditorText(0, role, child_parameters));
                } else if (type === "completion") {
                    this.addOperation(new OperationEditorCompletion(0, role, child_parameters));
                } else if (type === "branch") {
                    this.addOperation(new OperationEditorBranch(0, role, child_parameters));
                }
            }
        } else {
            const newChild = new OperationEditorText(0, "assistant");
            this.addOperation(newChild);
        }
    }

    setId(id) {
        this.id = id;
        this.setAttribute("data-id", id);
    }

    isValid() {
        for (const operation of this.operations) {
            if (!operation.valid) {
                return false;
            }
        }
        return true;
    }

    addOperation(operation, afterOperationId) {
        operation.onUpdate = () => {
            this.onUpdate();
        };
        operation.onAdd = (id, parent) => {
            this.onAdd(id, parent || this);
        };
        operation.onRemove = (id) => {
            this.removeOperation(id);
        };
        if (afterOperationId) {
            const index = this.operations.findIndex(operation => operation.id === afterOperationId) + 1;
            this.operations.splice(index, 0, operation);
            this.parameters.operations.splice(index, 0, [operation.type, operation.role, operation.parameters]);
            this.insertBefore(operation, this.children[index]);
        } else {
            this.operations.push(operation);
            this.parameters.operations.push([operation.type, operation.role, operation.parameters]);
            this.appendChild(operation);
        }
        this.onUpdate();
    }

    removeOperation(id) {
        const index = this.operations.findIndex(operation => operation.id === id);
        this.operations.splice(index, 1);
        this.parameters.operations.splice(index, 1);
        this.removeChild(this.children[index]);
        this.onUpdate();
    }
}

class OperationEditorBranch extends OperationEditor {
    constructor(id, defaultRole, params) {
        super(id, "branch", defaultRole, "operation-editor-branch", params || {
            "options": [],
        });

        this.options = [];

        if (this.parameters["options"].length > 0) {
            for (const option of this.parameters["options"]) {
                const childOperations = new OperationEditorChildOperations(0, option, () => {
                    this.onChildrenUpdate();
                }, (idx, parent) => {
                    this.onAdd(idx, parent);
                }, (idx) => {
                    this.onRemove(idx);
                });
                this.addOption(childOperations);
            }
        } else {
            const newOption = new OperationEditorChildOperations(0, {
                "operations": [],
            }, () => {
                this.onChildrenUpdate();
            }, (idx, parent) => {
                this.onAdd(idx, parent);
            }, (idx) => {
                this.onRemove(idx);
            });
            this.addOption(newOption);
        }

        this.querySelector(".add-branch").addEventListener("click", () => {
            const newOption = new OperationEditorChildOperations(0, {
                "operations": [],
            }, () => {
                this.onChildrenUpdate();
            }, (idx, parent) => {
                this.onAdd(idx, parent);
            }, (idx) => {
                this.onRemove(idx);
            });
            this.addOption(newOption);
            this.onChildrenUpdate();
        });
        this.querySelector(".remove-branch").addEventListener("click", () => {
            // Get the selected branch
            const selectedBranch = parseInt(this.querySelector("input[name=selected-branch]:checked").value);
            this.removeOption(selectedBranch - 1);
            this.onChildrenUpdate();
        });

        this.updateIds();
        this.onChildrenUpdate();
        this.checkValid();
    }

    onChildrenUpdate() {
        this.parameters["options"] = this.options.map(option => option.parameters);
        this.onUpdate();
    }

    updateIds() {
        for (let i = 0; i < this.options.length; i++) {
            this.options[i].setId(i + 1);
        }
        const selectorParent = this.querySelector(".branch-selector");
        for (let i = 0; i < selectorParent.children.length; i++) {
            selectorParent.children[i].querySelector("input").value = (i + 1).toString();
        }
        if (!this.querySelector("input[name=selected-branch]:checked")) {
            selectorParent.children[0].querySelector("input").checked = true;
        }
    }

    checkValid() {
        if (this.options === undefined) {
            this.valid = false;
        } else if (this.options.length === 0) {
            this.valid = false;
        } else {
            this.valid = true;
            for (const option of this.options) {
                if (!option.isValid()) {
                    this.valid = false;
                    break;
                }
            }
        }
        this.onUpdate();
    }

    forEachChild(callback) {
        for (const option of this.options) {
            for (const operation of option.operations) {
                callback(operation);
            }
        }
    }

    addOption(childOperations) {
        this.options.push(childOperations);
        this.appendChild(childOperations);

        const index = this.options.length;
        const selectorParent = this.querySelector(".branch-selector");
        const selector = document.createElement("label");
        selector.innerHTML = `<input type="radio" name="selected-branch" value="${index}" checked>${index}`;
        selectorParent.appendChild(selector);
        this.updateIds();
    }

    removeOption(index) {
        this.options[index].remove();
        this.options.splice(index, 1);
        const selectorParent = this.querySelector(".branch-selector");
        selectorParent.removeChild(selectorParent.children[index]);
        this.updateIds();
    }

    onSubmit() {
        if (!this.valid) {
            return null;
        }
        let forks = [];
        for (const option of this.options) {
            let optionForks = [];
            for (const operation of option.operations) {
                optionForks.push(operation.onSubmit());
            }
            forks.push(optionForks);
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
customElements.define("operation-editor-child-operations", OperationEditorChildOperations, { extends: "div" });
