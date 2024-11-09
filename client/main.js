class AppState {
    constructor() {
        this.operations = [];
        this.operationsByID = {};
        this.parentsByID = {};
        this.results = null;
        this.addOperationNextId = null;
        this.addOperationParent = null;
        const serialized = localStorage.getItem("state");
        if (serialized) {
            const state = JSON.parse(serialized);
            for (const [type, role, parameters] of state) {
                if (type === "text") {
                    this.addOperation(new OperationEditorText(this.operations.length + 1, role, parameters));
                } else if (type === "completion") {
                    this.addOperation(new OperationEditorCompletion(this.operations.length + 1, role, parameters));
                } else if (type === "branch") {
                    this.addOperation(new OperationEditorBranch(this.operations.length + 1, role, parameters));
                }
            }
        }

        if (this.operations.length === 0) {
            this.addOperation(new OperationEditorText(1, "system"));
            this.addOperation(new OperationEditorText(2, "user"));
            this.addOperation(new OperationEditorCompletion(3, "assistant"));
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
            this.onAddOperationDialogSubmit();
        });
    }

    addOperation(operation, afterOperationId) {
        operation.onUpdate = () => {
            this.onUpdate();
        };
        operation.onAdd = (id, parent) => {
            this.showAddOperationDialog(id, parent);
        };
        operation.onRemove = (id) => {
            this.removeOperation(id);
        };

        const mainOperations = document.getElementById("main-operations");
        if (afterOperationId) {
            const index = this.operations.findIndex(operation => operation.id === afterOperationId) + 1;
            this.operations.splice(index, 0, operation);
            mainOperations.insertBefore(operation, mainOperations.children[index]);
        } else {
            this.operations.push(operation);
            mainOperations.appendChild(operation);
        }
    }

    showAddOperationDialog(operationId, parent) {
        this.addOperationNextId = operationId;
        this.addOperationParent = parent;
        document.getElementById('add-operation-dialog').showModal();
    }

    onAddOperationDialogSubmit() {
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

        if (this.addOperationParent !== null) {
            this.addOperationParent.addOperation(operation, this.addOperationNextId);
        } else {
            this.addOperation(operation, this.addOperationNextId);
        }
        this.onUpdate();
    }

    removeOperation(operationId) {
        const operation = this.operationsByID[operationId];
        delete this.operationsByID[operationId];
        this.operations = this.operations.filter(operation => operation.id !== operationId);
        document.getElementById("main-operations").removeChild(operation);
        this.onUpdate();
    }

    onUpdate() {
        this.submitButton.disabled = !this.operations.every(operation => operation.valid);

        let nextOperationId = 1;
        this.operationsByID = {};
        this.parentsByID = {};
        const _renumberOperation = (operation, parent) => {
            operation.setId(nextOperationId);
            this.operationsByID[nextOperationId] = operation;
            this.parentsByID[nextOperationId] = parent;
            nextOperationId++;
            operation.forEachChild((childOperation) => _renumberOperation(childOperation, operation));
        };
        this.operations.forEach((operation) => _renumberOperation(operation, this));

        const serialized = this.operations.map(operation => [operation.type, operation.role, operation.parameters]);
        localStorage.setItem("state", JSON.stringify(serialized));
    }

    onSubmit() {
        let operations = [];
        for (const operation of this.operations) {
            operations.push(operation.onSubmit());
        }
        fetch("/completion", {
            method: "POST",
            body: JSON.stringify({ operations }),
        }).then(response => response.json()).then(data => {
            this.results = new Results(data);
            const results = document.getElementById("results");
            results.innerHTML = "";
            results.appendChild(this.results);
            this.results.onApplyEdits = (operationId, textParameters, branchParameters) => {
                this.onApplyEdits(operationId, textParameters, branchParameters);
            };
        });
    }

    onApplyEdits(operationId, textParameters, branchParameters) {
        const operation = this.operationsByID[operationId];
        let parent = this.parentsByID[operationId];

        const textOperation = new OperationEditorText(0, operation.role, textParameters);
        const branchOperation = new OperationEditorBranch(0, operation.role, branchParameters);

        parent.addOperation(branchOperation, operationId);
        parent.addOperation(textOperation, operationId);
        parent.removeOperation(operationId);
    }

}

document.addEventListener("DOMContentLoaded", () => {
    new AppState();
});
