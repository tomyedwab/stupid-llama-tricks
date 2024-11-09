class AppState {
    constructor() {
        this.blocks = [];
        this.blocksByID = {};
        this.results = null;
        this.addBlockNextId = null;
        this.addBlockParent = null;
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

        document.getElementById('add-operation-dialog').querySelector('button.ok').addEventListener("click", () => {
            this.onAddBlockDialogSubmit();
        });
    }

    addBlock(block, afterBlockId) {
        block.onUpdate = () => {
            this.onUpdate();
        };
        block.onAdd = (id, parent) => {
            this.onAddBlock(id, parent);
        };
        block.onRemove = (id) => {
            this.onRemoveBlock(id);
        };

        const conversationBlocks = document.getElementById("conversation-blocks");
        if (afterBlockId) {
            const index = this.blocks.findIndex(block => block.id === afterBlockId) + 1;
            this.blocks.splice(index, 0, block);
            conversationBlocks.insertBefore(block, conversationBlocks.children[index]);
        } else {
            this.blocks.push(block);
            conversationBlocks.appendChild(block);
        }
        this.blocksByID[block.id] = block;
    }

    onAddBlock(blockId, parent) {
        this.addBlockNextId = blockId;
        this.addBlockParent = parent;
        document.getElementById('add-operation-dialog').showModal();
    }

    onAddBlockDialogSubmit() {
        const dialog = document.getElementById('add-operation-dialog');
        const type = dialog.querySelector("input[name=type]:checked").value;

        let operation;
        if (type === "text") {
            operation = new OperationEditorText(0, "assistant");
        } else if (type === "completion") {
            operation = new OperationEditorCompletion(0, "assistant");
        } else if (type === "branch") {
            operation = new OperationEditorBranch(0, "assistant");
        }

        if (this.addBlockParent !== null) {
            this.addBlockParent.addOperation(operation, this.addBlockNextId);
        } else {
            this.addBlock(operation, this.addBlockNextId);
        }
        this.onUpdate();
    }

    onRemoveBlock(blockId) {
        const block = this.blocksByID[blockId];
        delete this.blocksByID[blockId];
        this.blocks = this.blocks.filter(block => block.id !== blockId);
        document.getElementById("conversation-blocks").removeChild(block);
        this.onUpdate();
    }

    onUpdate() {
        this.submitButton.disabled = !this.blocks.every(block => block.valid);

        let nextBlockId = 1;
        this.blocksById = {};
        const _renumberBlock = (block) => {
            block.setId(nextBlockId);
            this.blocksByID[nextBlockId] = block;
            nextBlockId++;
            block.forEachChild(_renumberBlock);
        };
        this.blocks.forEach(_renumberBlock);

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
