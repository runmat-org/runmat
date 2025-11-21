import * as vscode from 'vscode';
import {
  Executable,
  LanguageClient,
  LanguageClientOptions,
  State,
  ServerOptions,
  TransportKind,
} from 'vscode-languageclient/node';

let client: LanguageClient | undefined;
let statusItem: vscode.StatusBarItem | undefined;
let outputChannel: vscode.OutputChannel | undefined;

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

  client = new LanguageClient('runmatLanguageServer', 'RunMat Language Server', serverOptions, clientOptions);

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
  statusItem?.dispose();
  outputChannel?.dispose();
}
