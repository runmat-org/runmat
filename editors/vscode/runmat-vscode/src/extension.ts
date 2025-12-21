import * as vscode from 'vscode';
import * as path from 'path';
import {
  Executable,
  LanguageClient,
  LanguageClientOptions,
  State,
  ServerOptions,
  TransportKind,
  PositionEncodingKind,
} from 'vscode-languageclient/node';

let client: LanguageClient | undefined;
let statusItem: vscode.StatusBarItem | undefined;
let outputChannel: vscode.OutputChannel | undefined;
let runmatTerminal: vscode.Terminal | undefined;

function resolveServerExecutable(): Executable {
  const config = vscode.workspace.getConfiguration('runmat');
  const command = config.get<string>('lsp.path', 'runmat-lsp');
  const args = config.get<string[]>('lsp.extraArgs', []);

  return {
    command,
    args,
    transport: TransportKind.stdio,
    options: {
      env: {
        ...process.env,
      },
    },
  };
}

async function startLanguageClient(context: vscode.ExtensionContext): Promise<void> {
  const serverExecutable = resolveServerExecutable();

  const clientOptions: LanguageClientOptions = {
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

  const serverOptions: ServerOptions = serverExecutable;

  client = new LanguageClient(
    'runmatLanguageServer',
    'RunMat Language Server',
    serverOptions,
    clientOptions,
  );
  
  // Register proposed features to support UTF-8 position encoding
  client.registerProposedFeatures();

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
  } catch (error) {
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
    if (event.newState === State.Running) {
      statusItem.text = 'RunMat LS: Running';
    } else if (event.newState === State.Stopped) {
      statusItem.text = 'RunMat LS: Stopped';
    }
  });

  client.onNotification('runmat/status', (payload: { message?: string; tooltip?: string }) => {
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

async function stopLanguageClient(): Promise<void> {
  if (!client) {
    return;
  }
  const currentClient = client;
  client = undefined;
  try {
    await currentClient.stop();
  } catch (error) {
    outputChannel?.appendLine(`[RunMat] Error while stopping language server: ${error}`);
  }
}

function getOrCreateTerminal(): vscode.Terminal {
  if (runmatTerminal && runmatTerminal.exitStatus === undefined) {
    return runmatTerminal;
  }

  const runmatBinary = findRunmatBinary();
  if (!runmatBinary) {
    throw new Error('runmat binary not found. Build with: cargo build --release --bin runmat');
  }

  // Create terminal with runmat REPL
  runmatTerminal = vscode.window.createTerminal({
    name: 'RunMat',
    shellPath: runmatBinary,
    shellArgs: [],
    hideFromUser: false,
  });

  return runmatTerminal;
}

function findRunmatBinary(): string | null {
  const config = vscode.workspace.getConfiguration('runmat');
  const configuredPath = config.get<string>('executable.path');
  
  if (configuredPath) {
    return configuredPath;
  }

  // Try workspace root
  const workspaceFolder = vscode.workspace.workspaceFolders?.[0];
  if (workspaceFolder) {
    const candidates = [
      path.join(workspaceFolder.uri.fsPath, 'target/release/runmat.exe'),
      path.join(workspaceFolder.uri.fsPath, 'target/release/runmat'),
    ];
    for (const candidate of candidates) {
      const fs = require('fs');
      if (fs.existsSync(candidate)) {
        return candidate;
      }
    }
  }

  // Try system PATH
  const systemPaths = ['runmat.exe', 'runmat'];
  return systemPaths[0]; // Return first, system will resolve from PATH
}

async function runFile(filePath: string): Promise<void> {
  try {
    const terminal = getOrCreateTerminal();
    terminal.show();
    
    // Send run command to terminal
    const escapedPath = filePath.replace(/\\/g, '/');
    terminal.sendText(`run('${escapedPath}')`, true);
  } catch (error) {
    vscode.window.showErrorMessage(`Failed to run file: ${error}`);
  }
}

async function runSelection(editor: vscode.TextEditor): Promise<void> {
  try {
    const { selection, document } = editor;
    const code = document.getText(selection);
    
    if (!code.trim()) {
      vscode.window.showWarningMessage('No code selected');
      return;
    }

    const terminal = getOrCreateTerminal();
    terminal.show();

    // Send selected code to terminal
    // For multi-line code, we need to handle it carefully
    const lines = code.split('\n');
    for (const line of lines) {
      if (line.trim()) {
        terminal.sendText(line, false);
      }
    }
    terminal.sendText('', true); // Final enter to execute
  } catch (error) {
    vscode.window.showErrorMessage(`Failed to run selection: ${error}`);
  }
}

function autoIndent(editor: vscode.TextEditor): void {
  try {
    const { document, selection } = editor;
    
    // Get range to format: selection or entire document
    let range: vscode.Range;
    if (selection.isEmpty) {
      range = new vscode.Range(
        document.positionAt(0),
        document.positionAt(document.getText().length)
      );
    } else {
      range = new vscode.Range(selection.start.line, 0, selection.end.line, document.lineAt(selection.end.line).text.length);
    }

    const lines = document.getText(range).split('\n');
    const indentedLines: string[] = [];
    let indentLevel = 0;
    const indentSize = 4; // Standard MATLAB indent

    for (const line of lines) {
      const trimmed = line.trim();
      
      if (!trimmed) {
        indentedLines.push('');
        continue;
      }

      // Decrease indent for 'end', 'else', 'elseif', 'catch', 'otherwise'
      if (/^\b(end|else|elseif|catch|otherwise)\b/i.test(trimmed)) {
        indentLevel = Math.max(0, indentLevel - 1);
      }

      // Apply current indent level
      const indent = ' '.repeat(indentLevel * indentSize);
      indentedLines.push(indent + trimmed);

      // Increase indent after 'function', 'if', 'for', 'while', 'switch', 'try', 'class'
      if (/\b(function|if|for|while|switch|try|classdef|methods|properties)\b/i.test(trimmed) && !trimmed.endsWith('end')) {
        indentLevel++;
      }
    }

    // Apply changes
    editor.edit((editBuilder) => {
      editBuilder.replace(range, indentedLines.join('\n'));
    });
  } catch (error) {
    vscode.window.showErrorMessage(`Failed to auto-indent: ${error}`);
  }
}

function toggleComment(editor: vscode.TextEditor, comment: boolean): void {
  try {
    const { document, selection } = editor;
    
    // Get range: selection or current line
    let startLine = selection.start.line;
    let endLine = selection.isEmpty ? selection.start.line : selection.end.line;
    
    editor.edit((editBuilder) => {
      for (let lineNum = startLine; lineNum <= endLine; lineNum++) {
        const line = document.lineAt(lineNum);
        const text = line.text;
        const trimmed = text.trim();
        
        if (!trimmed) continue;
        
        if (comment) {
          // Add comment
          const indent = text.match(/^(\s*)/)?.[1] || '';
          editBuilder.replace(line.range, indent + '%' + (trimmed[0] !== '%' ? ' ' : '') + trimmed);
        } else {
          // Remove comment
          if (trimmed.startsWith('%')) {
            const indent = text.match(/^(\s*)/)?.[1] || '';
            let uncommented = trimmed.substring(1);
            if (uncommented.startsWith(' ')) {
              uncommented = uncommented.substring(1);
            }
            editBuilder.replace(line.range, indent + uncommented);
          }
        }
      }
    });
  } catch (error) {
    vscode.window.showErrorMessage(`Failed to toggle comment: ${error}`);
  }
}

function duplicateLines(editor: vscode.TextEditor): void {
  try {
    const { document, selection } = editor;
    
    const startLine = selection.start.line;
    const endLine = selection.isEmpty ? selection.start.line : selection.end.line;
    
    editor.edit((editBuilder) => {
      for (let i = endLine; i >= startLine; i--) {
        const line = document.lineAt(i);
        editBuilder.insert(line.range.end, '\n' + line.text);
      }
    });
  } catch (error) {
    vscode.window.showErrorMessage(`Failed to duplicate lines: ${error}`);
  }
}

function changeCase(editor: vscode.TextEditor, toUpperCase: boolean): void {
  try {
    const { document, selection } = editor;
    
    if (selection.isEmpty) {
      vscode.window.showWarningMessage('Please select text to change case');
      return;
    }

    const selectedText = document.getText(selection);
    const newText = toUpperCase ? selectedText.toUpperCase() : selectedText.toLowerCase();
    
    editor.edit((editBuilder) => {
      editBuilder.replace(selection, newText);
    });
  } catch (error) {
    vscode.window.showErrorMessage(`Failed to change case: ${error}`);
  }
}

export async function activate(context: vscode.ExtensionContext): Promise<void> {
  outputChannel = vscode.window.createOutputChannel('RunMat Language Server');
  statusItem = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Left);
  statusItem.name = 'RunMat Language Server';
  statusItem.text = 'RunMat LS: Initializing…';
  statusItem.tooltip = 'Starting the RunMat language server';
  statusItem.show();
  context.subscriptions.push(outputChannel, statusItem);

  context.subscriptions.push(
    vscode.commands.registerCommand('runmat.restartServer', async () => {
      outputChannel?.appendLine('[RunMat] Restarting language server…');
      await stopLanguageClient();
      await startLanguageClient(context);
    }),
    vscode.commands.registerCommand('runmat.showLogs', () => {
      outputChannel?.show(true);
    }),
    vscode.commands.registerCommand('runmat.runFile', async () => {
      const editor = vscode.window.activeTextEditor;
      if (!editor) {
        vscode.window.showErrorMessage('No active editor');
        return;
      }
      await runFile(editor.document.uri.fsPath);
    }),
    vscode.commands.registerCommand('runmat.runSelection', async () => {
      const editor = vscode.window.activeTextEditor;
      if (!editor) {
        vscode.window.showErrorMessage('No active editor');
        return;
      }
      await runSelection(editor);
    }),
    vscode.commands.registerCommand('runmat.clearTerminal', async () => {
      if (runmatTerminal) {
        runmatTerminal.sendText('clear', true);
      }
    }),
    vscode.commands.registerCommand('runmat.autoIndent', () => {
      const editor = vscode.window.activeTextEditor;
      if (!editor) {
        vscode.window.showErrorMessage('No active editor');
        return;
      }
      autoIndent(editor);
    }),
    vscode.commands.registerCommand('runmat.commentLines', () => {
      const editor = vscode.window.activeTextEditor;
      if (!editor) {
        vscode.window.showErrorMessage('No active editor');
        return;
      }
      toggleComment(editor, true);
    }),
    vscode.commands.registerCommand('runmat.uncommentLines', () => {
      const editor = vscode.window.activeTextEditor;
      if (!editor) {
        vscode.window.showErrorMessage('No active editor');
        return;
      }
      toggleComment(editor, false);
    }),
    vscode.commands.registerCommand('runmat.duplicateLines', () => {
      const editor = vscode.window.activeTextEditor;
      if (!editor) {
        vscode.window.showErrorMessage('No active editor');
        return;
      }
      duplicateLines(editor);
    }),
    vscode.commands.registerCommand('runmat.changeCaseUpper', () => {
      const editor = vscode.window.activeTextEditor;
      if (!editor) {
        vscode.window.showErrorMessage('No active editor');
        return;
      }
      changeCase(editor, true);
    }),
    vscode.commands.registerCommand('runmat.changeCaseLower', () => {
      const editor = vscode.window.activeTextEditor;
      if (!editor) {
        vscode.window.showErrorMessage('No active editor');
        return;
      }
      changeCase(editor, false);
    }),
    vscode.workspace.onDidChangeConfiguration(async (event) => {
      if (event.affectsConfiguration('runmat.lsp.path') || event.affectsConfiguration('runmat.lsp.extraArgs')) {
        outputChannel?.appendLine('[RunMat] Configuration changed, restarting language server…');
        await stopLanguageClient();
        await startLanguageClient(context);
      }
    })
  );

  await startLanguageClient(context);
}

export async function deactivate(): Promise<void> {
  await stopLanguageClient();
  
  // Kill RunMat terminal if running
  if (runmatTerminal && runmatTerminal.exitStatus === undefined) {
    runmatTerminal.dispose();
  }
  
  statusItem?.dispose();
  outputChannel?.dispose();
}
