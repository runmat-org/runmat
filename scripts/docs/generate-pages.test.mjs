// The committed Validation pages, Status and the procedure table equal what the generator writes
// from the ledger, the evidence catalog and the example receipts.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";
import { generatedPages, pageIsCurrent } from "./generate-pages.mjs";
import { maskTestCounts, testFamily } from "./test-counts.mjs";

const source = resolve(dirname(fileURLToPath(import.meta.url)), "../..");

test("committed generated pages match the generator", () => {
    for (const [path, text] of Object.entries(generatedPages(source)))
        assert.ok(pageIsCurrent(path, readFileSync(join(source, path), "utf8"), text), `${path} is stale; run node scripts/docs/generate-pages.mjs`);
});

test("every family page shows reference checks or names the planned one", () => {
    for (const [path, text] of Object.entries(generatedPages(source))) {
        if (!/validation\/(?!status)[a-z-]+\.md$/.test(path)) continue;
        assert.ok(/^### /m.test(text) || /^No reference check yet\. Planned: .+\.$/m.test(text), `${path} has no check and no planned check`);
        assert.ok(!text.includes("| Capability |"), `${path} still carries a capability table`);
    }
});

test("Status counts automated tests per family and masks them for the staleness check", () => {
    const status = generatedPages(source)["docs/simulation/validation/status.md"];
    assert.match(status, /\| Reference checks \| Automated tests \|/);
    const structural = status.match(/^\| \[Structural mechanics\][^\n]*\| \d+ \| ([\d,]+) \|/m);
    assert.ok(structural && Number(structural[1].replace(/,/g, "")) > 100, "structural automated-test count is missing or implausibly small");
    assert.equal(maskTestCounts(status.replace(/\| ([\d,]+) \| (\d{4}-|none)/, "| 1 | $2")), maskTestCounts(status));
});

test("tests are attributed to coupled families before their parts", () => {
    assert.equal(testFamily("series_conduction_through_solid_and_fluid", "crates/runmat-analysis/fea/src/physics/conjugate_heat/tests"), "conjugate-heat-transfer");
    assert.equal(testFamily("inlet_velocity_reproduces_couette_flow", "crates/runmat-analysis/fea/src/physics/cfd/canonical/tests"), "fluid");
    assert.equal(testFamily("workflow_retains_dependencies", "crates/runmat-analysis/execution/src/tests"), "shared");
});
