// The committed Validation pages, Status and the procedure table equal what the generator writes
// from the ledger, the evidence catalog and the example receipts.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";
import { generatedPages } from "./generate-pages.mjs";

const source = resolve(dirname(fileURLToPath(import.meta.url)), "../..");

test("committed generated pages match the generator", () => {
    for (const [path, text] of Object.entries(generatedPages(source)))
        assert.equal(readFileSync(join(source, path), "utf8"), text, `${path} is stale; run node scripts/docs/generate-pages.mjs`);
});

test("every family page shows reference checks or names the planned one", () => {
    for (const [path, text] of Object.entries(generatedPages(source))) {
        if (!/validation\/(?!status)[a-z-]+\.md$/.test(path)) continue;
        assert.ok(/^### /m.test(text) || /^No reference check yet\. Planned: .+\.$/m.test(text), `${path} has no check and no planned check`);
        assert.ok(!text.includes("| Capability |"), `${path} still carries a capability table`);
    }
});
