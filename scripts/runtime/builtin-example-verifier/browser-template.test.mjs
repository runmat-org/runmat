import assert from "node:assert/strict";
import test from "node:test";

import { createRunnerHtml } from "./browser-template.mjs";

test("browser setup consumes only the declared fixture payload", () => {
    const html = createRunnerHtml(1_000, 1, 250, false);
    assert.match(html, /testCase\.fixture/);
    assert.match(html, /fixture\.Filesystem\?\.entries/);
    assert.match(html, /fixture\.ApplicationSettings \? .*isolatedMemory/);
    assert.match(html, /session\.dispose\(\);/);
    assert.match(html, /session\.free\(\);/);
    assert.match(html, /session\.dispose\(\);[\s\S]*?catch \(err\) \{[\s\S]*?if \(!errorText\)[\s\S]*?\} finally \{[\s\S]*?session\.free\(\);/);
    assert.doesNotMatch(html, /input\.includes/);
    assert.doesNotMatch(html, /solver\.m/);
    assert.doesNotMatch(html, /fixtures\/high_ascii\.txt/);
});

test("worker cleanup frees after dispose failure without erasing the execution error", () => {
    const html = createRunnerHtml(1_000, 1, 250, false);
    const encoded = html.match(/const workerSource = ("(?:\\.|[^"\\])*");/);
    assert.ok(encoded, "worker source is embedded as a JSON string");
    const source = JSON.parse(encoded[1]);
    const begin = source.indexOf("    if (session) {");
    const end = source.indexOf("\n  }\n  self.postMessage", begin);
    assert.ok(begin >= 0 && end > begin, "worker has a bounded terminal cleanup block");
    const cleanup = new Function("session", "primaryError", `let errorText = primaryError;\n${source.slice(begin, end)}\nreturn errorText;`);
    let frees = 0;
    const session = { dispose() { throw new Error("dispose failed"); }, free() { frees += 1; } };
    assert.equal(cleanup(session, "primary execution failure"), "primary execution failure");
    assert.equal(frees, 1);
    assert.equal(cleanup(session, ""), "dispose failed");
    assert.equal(frees, 2);
});
