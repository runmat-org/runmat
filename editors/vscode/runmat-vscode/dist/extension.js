"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
exports.activate = activate;
exports.deactivate = deactivate;
const vscode = __importStar(require("vscode"));
const node_1 = require("vscode-languageclient/node");
let client;
let statusItem;
let outputChannel;
function resolveServerExecutable() {
    const config = vscode.workspace.getConfiguration('runmat');
    const command = config.get('lsp.path', 'runmat-lsp');
    const args = config.get('lsp.extraArgs', []);
    return {
        command,
        args,
        transport: node_1.TransportKind.stdio,
        options: {
            env: {
                ...process.env,
            },
        },
    };
}
async function startLanguageClient(context) {
    const serverExecutable = resolveServerExecutable();
    const clientOptions = {
        documentSelector: [
            { scheme: 'file', language: 'runmat' },
            { scheme: 'untitled', language: 'runmat' },
        ],
        synchronize: {
            configurationSection: 'runmat',
            fileEvents: vscode.workspace.createFileSystemWatcher('**/*.m'),
        },
        outputChannel,
    };
    const serverOptions = serverExecutable;
    client = new node_1.LanguageClient('runmatLanguageServer', 'RunMat Language Server', serverOptions, clientOptions);
    statusItem?.show();
    if (statusItem) {
        statusItem.text = 'RunMat LS: Starting…';
    }
    try {
        await client.start();
        if (statusItem) {
            statusItem.text = 'RunMat LS: Running';
            statusItem.tooltip = 'RunMat language features are available.';
        }
    }
    catch (error) {
        if (statusItem) {
            statusItem.text = 'RunMat LS: Failed';
            statusItem.tooltip = 'Failed to start the RunMat language server.';
        }
        outputChannel?.appendLine(`[RunMat] Failed to start language server: ${error}`);
        throw error;
    }
    client.onDidChangeState((event) => {
        if (!statusItem) {
            return;
        }
        if (event.newState === node_1.State.Running) {
            statusItem.text = 'RunMat LS: Running';
        }
        else if (event.newState === node_1.State.Stopped) {
            statusItem.text = 'RunMat LS: Stopped';
        }
    });
    client.onNotification('runmat/status', (payload) => {
        if (!statusItem) {
            return;
        }
        if (payload?.message) {
            statusItem.text = payload.message;
        }
        if (payload?.tooltip) {
            statusItem.tooltip = payload.tooltip;
        }
    });
}
async function stopLanguageClient() {
    if (!client) {
        return;
    }
    const currentClient = client;
    client = undefined;
    try {
        await currentClient.stop();
    }
    catch (error) {
        outputChannel?.appendLine(`[RunMat] Error while stopping language server: ${error}`);
    }
}
async function activate(context) {
    outputChannel = vscode.window.createOutputChannel('RunMat Language Server');
    statusItem = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Left);
    statusItem.name = 'RunMat Language Server';
    statusItem.text = 'RunMat LS: Initializing…';
    statusItem.tooltip = 'Starting the RunMat language server';
    statusItem.show();
    context.subscriptions.push(outputChannel, statusItem);
    context.subscriptions.push(vscode.commands.registerCommand('runmat.restartServer', async () => {
        outputChannel?.appendLine('[RunMat] Restarting language server…');
        await stopLanguageClient();
        await startLanguageClient(context);
    }), vscode.commands.registerCommand('runmat.showLogs', () => {
        outputChannel?.show(true);
    }), vscode.workspace.onDidChangeConfiguration(async (event) => {
        if (event.affectsConfiguration('runmat.lsp.path') || event.affectsConfiguration('runmat.lsp.extraArgs')) {
            outputChannel?.appendLine('[RunMat] Configuration changed, restarting language server…');
            await stopLanguageClient();
            await startLanguageClient(context);
        }
    }));
    await startLanguageClient(context);
}
async function deactivate() {
    await stopLanguageClient();
    statusItem?.dispose();
    outputChannel?.dispose();
}
//# sourceMappingURL=extension.js.map