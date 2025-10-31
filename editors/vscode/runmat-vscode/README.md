# RunMat VS Code Extension

The RunMat extension provides first-party language support powered by the `runmat-lsp`
language server. The extension integrates diagnostics, hover support, and automatic
workspace analysis for `.m` and `.runmat` files.

## Features

- Rich diagnostics sourced from the RunMat parser and high-level IR analysis, including
  hints when the type engine cannot infer a symbol.
- Type-aware hover cards for user symbols and builtins that surface RunMat type
  information, GPU/fusion tags, and deep links to the official documentation.
- Scope-sensitive completion for script variables, function locals, workspace globals,
  and the entire builtin catalogue.
- Document symbol provider to quickly navigate functions and class methods defined in a file.
- Status bar integration that reports the live analysis summary from the language server.
- Configurable language server path to integrate with custom toolchains and commands for
  restarting the server or opening its logs.

## Commands

- **RunMat: Restart Language Server** — restarts the background `runmat-lsp` process.
- **RunMat: Show Language Server Logs** — opens the language server output channel.

## Configuration

The following configuration options are available under the `RunMat` section:

- `runmat.lsp.path` — path to the `runmat-lsp` executable (defaults to `runmat-lsp` on the `PATH`).
- `runmat.lsp.extraArgs` — additional command line flags to provide when the server starts.

## Development

Install dependencies and compile the extension using:

```bash
npm install
npm run compile
```

Use `npm run watch` during development to continuously rebuild the extension.
