import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { appendFileSync, chmodSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL("../../", import.meta.url));
const host = "x86_64-unknown-linux-gnu";
const hasPowerShell = spawnSync("pwsh", ["-NoProfile", "-Command", "exit 0"]).status === 0;

for (const shell of ["bash", "pwsh"]) {
  for (const [profile, args] of [["release", []], ["dev", ["--profile", "dev"]], ["qualified", ["--profile=qualified"]]]) {
    test(`${shell} native build keeps ${profile} for archive, host packer and CLI`, {
      skip: shell === "bash" ? process.platform === "win32" : !hasPowerShell,
    }, () => {
      const dir = mkdtempSync(path.join(tmpdir(), "runmat-native-profile-"));
      try {
        const bin = path.join(dir, "bin"), log = path.join(dir, "commands.jsonl");
        mkdirSync(bin);
        mkdirSync(path.join(dir, "tmp"));
        const cargo = `#!/usr/bin/env node
const fs = require('node:fs'), path = require('node:path');
const args = process.argv.slice(2);
fs.appendFileSync(process.env.PROFILE_TEST_LOG, JSON.stringify(args) + '\\n');
const option = name => args[args.indexOf(name) + 1];
if (args[0] === 'rustc') {
  const profile = option('--profile'), target = args.includes('--target') ? option('--target') : '${host}';
  const out = path.join(process.env.CARGO_TARGET_DIR, target, profile === 'dev' ? 'debug' : profile);
  fs.mkdirSync(out, {recursive:true});
  fs.writeFileSync(path.join(out, 'librunmat_aot_runtime.a'), 'archive');
  console.log('note: native-static-libs: -lc');
} else if (args[0] === 'run') {
  for (const name of ['--payload-out', '--manifest-out', '--notices-out']) fs.writeFileSync(option(name), 'fixture');
}
`;
        for (const [name, source] of [["cargo", cargo], ["rustc", `#!/usr/bin/env node
console.log(process.argv.includes('-vV') ? 'rustc test\\nhost: ${host}' : 'librunmat_aot_runtime.a');
`]]) {
          const file = path.join(bin, name);
          writeFileSync(file, source);
          chmodSync(file, 0o755);
          if (process.platform === "win32") {
            appendFileSync(`${file}.cmd`, `@"${process.execPath}" "${file}" %*\r\n`);
          }
        }
        const env = { ...process.env, PATH: `${bin}${path.delimiter}${process.env.PATH}`,
          RUSTC: path.join(bin, process.platform === "win32" ? "rustc.cmd" : "rustc"),
          CARGO_TARGET_DIR: path.join(dir, "target"), RUNMAT_BUILD_TARGET: host,
          RUNMAT_BUILD_PROFILE: "release", PROFILE_TEST_LOG: log, TMPDIR: path.join(dir, "tmp") };
        // Node passes only one case-insensitive PATH entry to Windows children.
        if (process.platform === "win32") {
          for (const key of Object.keys(env)) if (key !== "PATH" && key.toUpperCase() === "PATH") delete env[key];
        }
        const script = path.join(root, `scripts/build-runmat-with-aot-runtime.${shell === "bash" ? "sh" : "ps1"}`);
        const command = shell === "bash" ? [script, ...args, "--locked", "--offline"]
          : ["-NoProfile", "-File", script, ...args, "--locked", "--offline"];
        const result = spawnSync(shell, command, { cwd: dir, env, encoding: "utf8" });
        assert.equal(result.status, 0, `${result.stdout}\n${result.stderr}`);
        const calls = readFileSync(log, "utf8").trim().split("\n").map(JSON.parse);
        assert.deepEqual(calls.map(call => call[0]), ["rustc", "run", "build"]);
        for (const call of calls) {
          assert.equal(call[call.indexOf("--profile") + 1], profile);
          assert(call.includes("--locked"));
          assert(call.includes("--offline"));
          assert.equal(call[call.indexOf("--target") + 1], host);
        }
        assert.equal(calls[1][calls[1].indexOf("--bin") + 1], "runmat-aot-pack");
      } finally {
        rmSync(dir, { recursive: true, force: true });
      }
    });
  }
}
