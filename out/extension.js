"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.deactivate = exports.activate = void 0;
// The module 'vscode' contains the VS Code extensibility API
// Import the module and reference it with the alias vscode in your code below
const vscode = require("vscode");
const cmd = require("child_process");
const fs = require("fs");
// this method is called when your extension is activated
// your extension is activated the very first time the command is executed
function activate(context) {
    const config = new Config(vscode.workspace.getConfiguration('buglocate'));
    console.log(config);
    // console.log(vscode.workspace.getConfiguration('buglocate').get('project.name'));
    context.subscriptions.push(vscode.commands.registerCommand('buglocate.init', () => {
        vscode.window.showInformationMessage('[buglocate] Start prepare for buglocate...');
        cmd.exec('.\\main.exe --doCollect --bugRepo ' + config.bugRepo + ' --product ' + config.product + ' --gitRepo ' + config.gitRepo, { cwd: 'E:\\buglocate\\src\\backend\\dist' }, (error, stdout, stderr) => {
            console.log(stdout, stderr);
            // console.log(stderr);
            // vscode.window.showInformationMessage('[buglocate] Collect finished.');
            cmd.exec('.\\main.exe --doMatch --bugRepo ' + config.bugRepo + ' --product ' + config.product + ' --gitRepo ' + config.gitRepo, { cwd: 'E:\\buglocate\\src\\backend\\dist' }, (error, stdout, stderr) => {
                console.log(stdout, stderr);
                // console.log(stderr);
                // vscode.window.showInformationMessage('[buglocate] Match finished.');
                cmd.exec('.\\main.exe --doMakeDataset --product ' + config.product + ' --gitRepo ' + config.gitRepo + ' --maxDatasetSize ' + config.maxDatasetSize, { cwd: 'E:\\buglocate\\src\\backend\\dist' }, (error, stdout, stderr) => {
                    console.log(stdout, stderr);
                    vscode.window.showInformationMessage('[buglocate] All done.');
                });
            });
        });
    }));
    vscode.commands.executeCommand('buglocate.init');
    context.subscriptions.push(vscode.commands.registerTextEditorCommand('buglocate.search', (textEditor) => {
        const query = getSelectedText(textEditor);
        fs.writeFile('E:/buglocate/src/backend/dist/queryfile.txt', query, (error) => {
            if (error !== null) {
                console.log(error);
            }
        });
        cmd.exec('.\\main.exe --doPredict --product "AspectJ" --query "queryfile.txt"', { cwd: 'E:\\buglocate\\src\\backend\\dist' }, (error, stdout, stderr) => {
            if (stderr !== null) {
                console.log(stderr);
            }
            vscode.window.createTreeView('bugLocateResult', {
                treeDataProvider: new LocateResultProvider(stdout)
            });
        });
    }));
    context.subscriptions.push(vscode.commands.registerCommand('buglocate.goToFile', (item) => {
        openLocalFile(config.gitRepo + '/' + String(item.description));
    }));
}
exports.activate = activate;
// this method is called when your extension is deactivated
function deactivate() { }
exports.deactivate = deactivate;
class Config {
    constructor(configuration) {
        // project
        this.product = configuration.get('project.name');
        this.bugRepo = configuration.get('project.bugRepo');
        this.gitRepo = configuration.get('project.gitRepo');
        // this.gitRepo = wo.uri.fsPath;  // TODO: only support open one project in a window.
        // dataset
        this.maxDatasetSize = configuration.get('dataset.maxDatasetSize');
        this.maxQueryLength = configuration.get('dataset.maxQueryLength');
        this.maxCodeK = configuration.get('dataset.maxCodeK');
        this.maxFileLine = configuration.get('dataset.maxFileLine');
        this.maxFuncLine = configuration.get('dataset.maxFuncLine');
        // score
        this.useCodeLength = configuration.get('score.useCodeLength');
        this.useTFIDF = configuration.get('score.useTFIDF');
        this.useLearning = configuration.get('score.useLearning');
        // level
        this.file = configuration.get('level.file');
        this.function = configuration.get('level.function');
    }
}
function getSelectedText(editor) {
    const activeDocument = editor.document;
    const selection = editor.selection;
    const { start, end } = selection;
    const selectText = activeDocument.getText(selection.with(start, end));
    return selectText;
}
function openLocalFile(filePath) {
    // 获取TextDocument对象
    vscode.workspace.openTextDocument(filePath)
        .then(doc => {
        // 在VSCode编辑窗口展示读取到的文本
        vscode.window.showTextDocument(doc);
    }, err => {
        console.log(`Open ${filePath} error, ${err}.`);
    }).then(undefined, err => {
        console.log(`Open ${filePath} error, ${err}.`);
    });
}
class LocateResultProvider {
    constructor(result) {
        this.result = result;
    }
    getTreeItem(element) {
        return element;
    }
    getChildren(element) {
        return Promise.resolve(getResultData(this.result));
    }
}
function getResultData(result) {
    const resultList = JSON.parse(result);
    let rl = new Array();
    for (let i = 0; i < resultList.length; i++) {
        let a = new Result(String(resultList[i][0]).slice(0, 5), resultList[i][1], vscode.TreeItemCollapsibleState.None);
        rl.push(a);
    }
    return rl;
}
class Result extends vscode.TreeItem {
    constructor(name, score, collapsibleState) {
        super(name, collapsibleState);
        this.name = name;
        this.score = score;
        this.collapsibleState = collapsibleState;
        this.description = this.score;
    }
}
//# sourceMappingURL=extension.js.map