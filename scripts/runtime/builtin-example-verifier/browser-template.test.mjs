import assert from "node:assert/strict";
import test from "node:test";

import { createRunnerHtml } from "./browser-template.mjs";

test("browser setup consumes only the declared fixture payload", () => {
    const html = createRunnerHtml(1_000, 1, 250, false);
    assert.match(html, /testCase\.fixture/);
    assert.match(html, /fixture\.Filesystem\.entries/);
    assert.doesNotMatch(html, /input\.includes/);
    assert.doesNotMatch(html, /solver\.m/);
    assert.doesNotMatch(html, /fixtures\/high_ascii\.txt/);
});
