class AppState {
    constructor() {
        this.blocks = [];
        this.blocksByID = {};
        this.results = null;

        const serialized = localStorage.getItem("state");
        if (serialized) {
            const state = JSON.parse(serialized);
            for (const [type, role, parameters] of state) {
                if (type === "text") {
                    this.addBlock(new OperationEditorText(this.blocks.length + 1, role, parameters));
                } else if (type === "completion") {
                    this.addBlock(new OperationEditorCompletion(this.blocks.length + 1, role, parameters));
                } else if (type === "branch") {
                    this.addBlock(new OperationEditorBranch(this.blocks.length + 1, role, parameters));
                }
            }
        }

        if (this.blocks.length === 0) {
            this.addBlock(new OperationEditorText(1, "system"));
            this.addBlock(new OperationEditorText(2, "user"));
            this.addBlock(new OperationEditorCompletion(3, "assistant"));
        }

        this.submitButton = document.getElementById("submit");
        this.submitButton.addEventListener("click", () => {
            this.onSubmit();
        });
        this.onUpdate();

        document.body.addEventListener("click", () => {
            if (this.results) {
                this.results.handleWordSelection(null, false);
            }
        });
    }

    addBlock(block) {
        block.onUpdate = () => {
            this.onUpdate();
        };
        this.blocks.push(block);
        this.blocksByID[block.id] = block;
        document.getElementById("conversation-blocks").appendChild(block);
    }

    onUpdate() {
        this.submitButton.disabled = !this.blocks.every(block => block.valid);
        const serialized = this.blocks.map(block => [block.type, block.role, block.parameters]);
        localStorage.setItem("state", JSON.stringify(serialized));
    }

    onSubmit() {
        let operations = [];
        for (const block of this.blocks) {
            operations.push(block.onSubmit());
        }
        fetch("/completion", {
            method: "POST",
            body: JSON.stringify({ operations }),
        }).then(response => response.json()).then(data => {
            this.results = new Results(data);
            const results = document.getElementById("results");
            results.innerHTML = "";
            results.appendChild(this.results);
        });
    }

}

document.addEventListener("DOMContentLoaded", () => {
    new AppState();
});
