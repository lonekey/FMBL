// The module 'vscode' contains the VS Code extensibility API
// Import the module and reference it with the alias vscode in your code below
import * as vscode from 'vscode';
import * as cmd from 'child_process';
import * as fs from 'fs';
// this method is called when your extension is activated
// your extension is activated the very first time the command is executed
export function activate(context: vscode.ExtensionContext) {
	const config  = new Config(vscode.workspace.getConfiguration('buglocate'));
	console.log(config);
	// console.log(vscode.workspace.getConfiguration('buglocate').get('project.name'));

	context.subscriptions.push(vscode.commands.registerCommand('buglocate.init', () => {
		vscode.window.showInformationMessage('[buglocate] Start prepare for buglocate...');
		cmd.exec('.\\main.exe --doCollect --bugRepo '+config.bugRepo+' --product '+config.product+' --gitRepo '+config.gitRepo, {cwd:'E:\\buglocate\\src\\backend\\dist'}, (error, stdout, stderr) => {
			console.log(stdout, stderr);
			// console.log(stderr);
			// vscode.window.showInformationMessage('[buglocate] Collect finished.');
			cmd.exec('.\\main.exe --doMatch --bugRepo '+config.bugRepo+' --product '+config.product+' --gitRepo '+config.gitRepo, {cwd:'E:\\buglocate\\src\\backend\\dist'}, (error, stdout, stderr) => {
				console.log(stdout, stderr);
				// console.log(stderr);
				// vscode.window.showInformationMessage('[buglocate] Match finished.');
				cmd.exec('.\\main.exe --doMakeDataset --product '+config.product+' --gitRepo '+config.gitRepo+' --maxDatasetSize '+config.maxDatasetSize, {cwd:'E:\\buglocate\\src\\backend\\dist'}, (error, stdout, stderr) => {
					console.log(stdout, stderr);
					vscode.window.showInformationMessage('[buglocate] All done.');
				});
			});
		});
	}));
	vscode.commands.executeCommand('buglocate.init');
	context.subscriptions.push(vscode.commands.registerTextEditorCommand('buglocate.search', (textEditor) => {
		const query = getSelectedText(textEditor);
		fs.writeFile('E:/buglocate/src/backend/dist/queryfile.txt', query , (error) => {
			if (error !== null){				
				console.log(error);
			}
		});
		cmd.exec('.\\main.exe --doPredict --product "AspectJ" --query "queryfile.txt"', {cwd:'E:\\buglocate\\src\\backend\\dist'}, (error, stdout, stderr) => {
			if (stderr !== null){				
				console.log(stderr);
			}
			vscode.window.createTreeView('bugLocateResult', {
				treeDataProvider: new LocateResultProvider(stdout)
			  });
		});
	}));
	context.subscriptions.push(vscode.commands.registerCommand('buglocate.goToFile',(item: vscode.TreeItem) => {
		openLocalFile(config.gitRepo+'/'+String(item.description));
	}));
}

// this method is called when your extension is deactivated
export function deactivate() {}


class Config{
	product: any;
	bugRepo: any;
	gitRepo: any;
	maxDatasetSize: any;
	maxQueryLength: any;
	maxCodeK: any;
	maxFileLine: any;
	maxFuncLine: any;
	useCodeLength: any;
	useTFIDF: any;
	useLearning: any;
	file: any;
	function: any;

	constructor(configuration: vscode.WorkspaceConfiguration){
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

function getSelectedText(editor: vscode.TextEditor) {
	const activeDocument = editor.document;
	const selection = editor.selection;
	const { start, end } = selection;
	const selectText = activeDocument.getText(selection.with(start, end));
	return selectText;
}

function openLocalFile(filePath: string) {
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

class LocateResultProvider implements vscode.TreeDataProvider<Result> {
	constructor(private result: string){}
	getTreeItem(element: Result): vscode.TreeItem {
	  return element;
	}
  
	getChildren(element?: Result): Thenable<Result[]> {
		return Promise.resolve(getResultData(this.result));
  }
}
  
  
function getResultData(result: string): Result[]{
	const resultList = JSON.parse(result);
	let rl :Result[] = new Array();
	for (let i =0;i< resultList.length; i++){
		let a = new Result(String(resultList[i][0]).slice(0,5), resultList[i][1], vscode.TreeItemCollapsibleState.None);
		rl.push(a);
	}
	return rl;
}
class Result extends vscode.TreeItem {
	constructor(
	  public readonly name: string,
	  private score: string,
	  public readonly collapsibleState: vscode.TreeItemCollapsibleState
	) {
	  super(name, collapsibleState);
	  this.description = this.score;
	}
  }