"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.NodeDependenciesProvider = void 0;
const vscode = require("vscode");
class NodeDependenciesProvider {
    getTreeItem(element) {
        return element;
    }
    getChildren(element) {
        if (element) {
            return Promise.resolve([new Result('name', "score", vscode.TreeItemCollapsibleState.Collapsed)]);
        }
        else {
            return Promise.resolve([]);
        }
    }
}
exports.NodeDependenciesProvider = NodeDependenciesProvider;
class Result extends vscode.TreeItem {
    constructor(name, score, collapsibleState) {
        super(name, collapsibleState);
        this.name = name;
        this.score = score;
        this.collapsibleState = collapsibleState;
        this.description = this.score;
    }
}
//# sourceMappingURL=fileExplorer.js.map