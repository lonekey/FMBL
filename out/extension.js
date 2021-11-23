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
    const configString = JSON.stringify(config);
    fs.writeFile(config.workpath + '/config.json', configString, (error) => {
        if (error !== null) {
            console.log(error);
        }
    });
    // console.log(configString);
    // console.log(vscode.workspace.getConfiguration('buglocate').get('project.name'));
    // downloadExe();
    context.subscriptions.push(vscode.commands.registerCommand('buglocate.init', () => {
        vscode.window.showInformationMessage('[buglocate] Start prepare for buglocate...');
        cmd.exec('.\\main.exe --doCollect', { cwd: config.workpath }, (error, stdout, stderr) => {
            console.log(stdout, stderr);
            // console.log(stderr);
            // vscode.window.showInformationMessage('[buglocate] Collect finished.');
            cmd.exec('.\\main.exe --doMatch', { cwd: config.workpath }, (error, stdout, stderr) => {
                console.log(stdout, stderr);
                // console.log(stderr);
                // vscode.window.showInformationMessage('[buglocate] Match finished.');
                cmd.exec('.\\main.exe --doMakeDataset', { cwd: config.workpath }, (error, stdout, stderr) => {
                    console.log(stdout, stderr);
                    if (config.useLearning) {
                        cmd.exec('.\\main.exe --doTrain', { cwd: config.workpath }, (error, stdout, stderr) => {
                            console.log(stdout, stderr);
                            vscode.window.showInformationMessage('[buglocate] All done.');
                        });
                    }
                    else {
                        vscode.window.showInformationMessage('[buglocate] All done.');
                    }
                });
            });
        });
    }));
    vscode.commands.executeCommand('buglocate.init');
    context.subscriptions.push(vscode.commands.registerTextEditorCommand('buglocate.search', (textEditor) => {
        vscode.window.setStatusBarMessage('locating...');
        const query = getSelectedText(textEditor);
        fs.writeFile(config.workpath + '/queryfile.txt', query, (error) => {
            if (error !== null) {
                console.log(error);
            }
        });
        cmd.exec('.\\main.exe --doPredict --query "queryfile.txt"', { cwd: config.workpath }, (error, stdout, stderr) => {
            if (stderr !== null) {
                console.log(stderr);
            }
            vscode.window.setStatusBarMessage('');
            vscode.window.createTreeView('bugLocateResult', {
                treeDataProvider: new LocateResultProvider(stdout)
            });
        });
    }));
    context.subscriptions.push(vscode.commands.registerCommand('buglocate.goToFile', (item) => {
        if (item.collapsibleState !== vscode.TreeItemCollapsibleState.None) {
            openLocalFile(config.gitRepo + '/' + String(item.description));
        }
    }));
}
exports.activate = activate;
// this method is called when your extension is deactivated
function deactivate() { }
exports.deactivate = deactivate;
function saveConfig(config) {
    console.log(JSON.stringify(config));
}
class Config {
    constructor(configuration) {
        this.workpath = configuration.get('workpath');
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
        //train
        this.learningRate = configuration.get('train.learningRate');
        this.batchSize = configuration.get('train.batchSize');
        this.epoch = configuration.get('train.epoch');
        this.negFileNum = configuration.get('train.negFileNum');
        this.negMethodNum = configuration.get('train.negMethodNum');
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
        this.resultList = JSON.parse(this.result);
    }
    getChildren(element) {
        if (element) {
            return Promise.resolve(getResultData(this.resultList[element.index]["methods"], "method"));
        }
        else {
            return Promise.resolve(getResultData(this.resultList, "file"));
        }
    }
    getTreeItem(element) {
        return element;
    }
}
function getResultData(resultList, type) {
    // const resultList = JSON.parse(result);
    console.log(resultList);
    let rl = new Array();
    for (let i = 0; i < resultList.length; i++) {
        if (type === "file") {
            let a = new Result(i, String(resultList[i]["score"]).slice(0, 5), resultList[i]["name"], vscode.TreeItemCollapsibleState.Collapsed, null, null);
            rl.push(a);
        }
        else {
            let a = new Result(i, String(resultList[i]["score"]).slice(0, 5), resultList[i]["name"], vscode.TreeItemCollapsibleState.None, resultList[i]["start"], resultList[i]["end"]);
            rl.push(a);
        }
    }
    return rl;
}
class Result extends vscode.TreeItem {
    constructor(index, score, name, collapsibleState, start, end) {
        super(score, collapsibleState);
        this.index = index;
        this.score = score;
        this.name = name;
        this.collapsibleState = collapsibleState;
        this.start = start;
        this.end = end;
        if (start === null || end === null) {
            this.description = name;
        }
        else {
            this.description = name + "  " + start[0] + "-" + end[0];
        }
        this.index = index;
    }
}
//# sourceMappingURL=extension.js.map