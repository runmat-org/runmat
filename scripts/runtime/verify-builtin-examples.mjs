#!/usr/bin/env node
// @ts-check

import { PROTOCOL_COMMANDS, runProtocolCli } from "./builtin-example-verifier/protocol-cli.mjs";

const command = process.argv[2];
if (PROTOCOL_COMMANDS.includes(command)) {
    await runProtocolCli(process.argv.slice(2));
} else {
    await import("./builtin-example-verifier/legacy-runner.mjs");
}
