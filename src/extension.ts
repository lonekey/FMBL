// The module 'vscode' contains the VS Code extensibility API
// Import the module and reference it with the alias vscode in your code below
import * as vscode from 'vscode';
import * as cmd from 'child_process';
import * as fs from 'fs';
import { download } from 'vscode-test';
// this method is called when your extension is activated
// your extension is activated the very first time the command is executed
export function activate(context: vscode.ExtensionContext) {
	const config  = new Config(vscode.workspace.getConfiguration('buglocate'));
	console.log(config);
	const configString = JSON.stringify(config);
	fs.writeFile(config.workpath+'/config.json', configString , (error) => {
		if (error !== null){				
			console.log(error);
		}
	});
	// console.log(configString);
	// console.log(vscode.workspace.getConfiguration('buglocate').get('project.name'));
	// downloadExe();
	context.subscriptions.push(vscode.commands.registerCommand('buglocate.init', () => {
		vscode.window.showInformationMessage('[buglocate] Start prepare for buglocate...');
		cmd.exec('.\\main.exe --doCollect', {cwd:config.workpath}, (error, stdout, stderr) => {
			console.log(stdout, stderr);
			// console.log(stderr);
			// vscode.window.showInformationMessage('[buglocate] Collect finished.');
			cmd.exec('.\\main.exe --doMatch', {cwd:config.workpath}, (error, stdout, stderr) => {
				console.log(stdout, stderr);
				// console.log(stderr);
				// vscode.window.showInformationMessage('[buglocate] Match finished.');
				cmd.exec('.\\main.exe --doMakeDataset', {cwd:config.workpath}, (error, stdout, stderr) => {
					console.log(stdout, stderr);
					if(config.useLearning){
						cmd.exec('.\\main.exe --doTrain', {cwd:config.workpath}, (error, stdout, stderr) => {
							console.log(stdout, stderr);
							vscode.window.showInformationMessage('[buglocate] All done.');
						});
					}else{
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
		fs.writeFile(config.workpath+'/queryfile.txt', query , (error) => {
			if (error !== null){				
				console.log(error);
			}
		});
		cmd.exec('.\\main.exe --doPredict --query "queryfile.txt"', {cwd:config.workpath}, (error, stdout, stderr) => {
			if (stderr !== null){				
				console.log(stderr);
			}
			vscode.window.setStatusBarMessage('');
			vscode.window.createTreeView('bugLocateResult', {
				treeDataProvider: new LocateResultProvider(stdout)
			  });
		});
	}));
	context.subscriptions.push(vscode.commands.registerCommand('buglocate.goToFile',(item: vscode.TreeItem) => {
		if(item.collapsibleState!==vscode.TreeItemCollapsibleState.None){
			openLocalFile(config.gitRepo+'/'+String(item.description));
		}
	}));
}

// this method is called when your extension is deactivated
export function deactivate() {}

function saveConfig(config: Config){
	console.log(JSON.stringify(config));


}


class Config{
	workpath: any;
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
	learningRate: any;
	batchSize: any;
	epoch: any;
	negFileNum: any;
	negMethodNum: any;


	constructor(configuration: vscode.WorkspaceConfiguration){
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
	resultList = JSON.parse(this.result);

	getChildren(element?: Result): Thenable<Result[]> {
		if(element){
			return Promise.resolve(getResultData(this.resultList[element.index]["methods"], "method"));
		}
		else{
			return Promise.resolve(getResultData(this.resultList,"file"));
		}
    }
	getTreeItem(element: Result): vscode.TreeItem {
	  return element;
	}
  

}
  
  
function getResultData(resultList: any, type: string): Result[]{
	// const resultList = JSON.parse(result);
	console.log(resultList);
	let rl :Result[] = new Array();
	for (let i =0;i< resultList.length; i++){
		if(type==="file"){
			let a = new Result(i, String(resultList[i]["score"]).slice(0,5), resultList[i]["name"], vscode.TreeItemCollapsibleState.Collapsed, null, null);
			rl.push(a);

		}else{
			let a = new Result(i, String(resultList[i]["score"]).slice(0,5), resultList[i]["name"], vscode.TreeItemCollapsibleState.None, resultList[i]["start"], resultList[i]["end"]);
			rl.push(a);
		}
	}
	return rl;
}
class Result extends vscode.TreeItem {
	constructor(
	  public index: any,
	  public readonly score: string,
	  public name: string,
	  public readonly collapsibleState: vscode.TreeItemCollapsibleState,
	  public start: any,
	  public end: any
	) {
	  super(score, collapsibleState);
	  if(start===null||end===null){
		this.description = name;
	  }else{
		this.description = name+"  "+start[0]+"-"+end[0];
	  }
	  
	  this.index = index;
	}
  }