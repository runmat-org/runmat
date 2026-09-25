import { spawnSync } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";

const MAX_OUTPUT_BYTES = 16 * 1024 * 1024;
const FILTER_TIMEOUT_MS = 120_000;

export function parseFocusedTestFilters(args) {
  if (args.length === 0 || args.length % 2 !== 0) {
    throw new Error("expected one or more --filter <Rust test path> pairs");
  }
  const filters = [];
  for (let index = 0; index < args.length; index += 2) {
    if (args[index] !== "--filter") throw new Error(`unexpected argument: ${args[index]}`);
    const filter = args[index + 1];
    if (!/^[A-Za-z_][A-Za-z0-9_#]*(::[A-Za-z_][A-Za-z0-9_#]*)*$/.test(filter)) {
      throw new Error(`invalid Rust test path: ${filter}`);
    }
    if (filters.includes(filter)) throw new Error(`duplicate Rust test path: ${filter}`);
    filters.push(filter);
  }
  return filters;
}

export function runFocusedTests(filters, run = spawnSync) {
  if (filters.length === 0) throw new Error("focused test selection is empty");
  let totalPassed = 0;
  const testedNames = new Set();
  for (const filter of filters) {
    const args = ["test", "-p", "runmat-runtime", "--lib", filter, "--", "--test-threads=1"];
    const result = run("cargo", args, {
      encoding: "utf8",
      maxBuffer: MAX_OUTPUT_BYTES,
      timeout: FILTER_TIMEOUT_MS,
    });
    if (result.error) throw new Error(`${filter}: Cargo did not complete: ${result.error.message}`);
    if (result.signal) throw new Error(`${filter}: test process ended by ${result.signal}`);
    if (result.status !== 0) {
      throw new Error(`${filter}: Cargo exited ${result.status}\n${result.stdout ?? ""}\n${result.stderr ?? ""}`);
    }
    const stdout = result.stdout ?? "";
    const summaries = [...stdout.matchAll(/^test result: ok\. (\d+) passed; 0 failed; (\d+) ignored;/gm)];
    if (summaries.length !== 1) throw new Error(`${filter}: expected one successful Rust test summary`);
    const passed = Number(summaries[0][1]);
    if (!Number.isSafeInteger(passed) || passed === 0) {
      throw new Error(`${filter}: no tests matched the reviewed filter`);
    }
    if (Number(summaries[0][2]) !== 0) throw new Error(`${filter}: selected tests were ignored`);
    const names = [...stdout.matchAll(/^test (.+) \.\.\. ok$/gm)].map((match) => match[1]);
    if (names.length !== passed) throw new Error(`${filter}: test-name inventory differs from the Rust summary`);
    for (const name of names) {
      if (testedNames.has(name)) throw new Error(`${filter}: test ${name} was selected by multiple filters`);
      testedNames.add(name);
      process.stdout.write(`passed ${name}\n`);
    }
    totalPassed += passed;
    process.stdout.write(`${filter}: ${passed} passed\n`);
  }
  process.stdout.write(`focused test groups: ${filters.length}; tests passed: ${totalPassed}\n`);
  return { groups: filters.length, passed: totalPassed, tests: [...testedNames] };
}

if (process.argv[1] && fileURLToPath(import.meta.url) === path.resolve(process.argv[1])) {
  try {
    runFocusedTests(parseFocusedTestFilters(process.argv.slice(2)));
  } catch (error) {
    process.stderr.write(`${error.message}\n`);
    process.exitCode = 1;
  }
}
