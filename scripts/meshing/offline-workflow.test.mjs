import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { spawnSync } from "node:child_process";
import test from "node:test";

for (const [file, prefix] of [["offline-meshing.yml", "offline-meshing-v2"], ["offline-qualification.yml", "offline-scientific-v2"]]) {
    const workflow = readFileSync(new URL(`../../.github/workflows/${file}`, import.meta.url), "utf8");
    const start = workflow.indexOf("      - name: Validate qualification source\n");
    assert.ok(start >= 0);
    const checkout = workflow.indexOf("      - uses: actions/checkout@", start);
    assert.ok(checkout > start);
    const validation = workflow.slice(start, checkout);
    const run = validation.indexOf("        run: |\n");
    assert.ok(run >= 0);
    const script = validation.slice(run + "        run: |\n".length)
        .split("\n").filter(line => line.trim()).map(line => {
            assert.ok(line.startsWith("          "));
            return line.slice(10);
        }).join("\n");

    test(`${file}: source admission requires the reviewed full SHA to equal the dispatch commit`, () => {
        const sha = "a".repeat(40);
        for (const [source, accepted] of [[sha, true], ["b".repeat(40), false], ["dev", false], ["", false]]) {
            const result = spawnSync("bash", ["-c", script], {
                encoding: "utf8",
                env: { PATH: process.env.PATH, EXPECTED_SOURCE: source, GITHUB_SHA: sha },
            });
            assert.equal(result.error, undefined);
            assert.equal(result.status === 0, accepted, `${source}: ${result.stderr}`);
        }
    });

    test(`${file}: checkout and caches use the workflow's identity rather than an independent source input`, () => {
        assert.match(validation, /EXPECTED_SOURCE: \$\{\{ inputs\.source_ref \}\}/);
        assert.equal(workflow.match(/inputs\.source_ref/g)?.length, 1, "source input only enters the validation environment");
        assert.match(workflow, /ref: \$\{\{ github\.sha \}\}\n\s+persist-credentials: false/);
        assert.ok(workflow.includes(`prefix-key: ${prefix}\n`));
        assert.match(workflow, /permissions:\n  contents: read\n/);
        assert.ok(workflow.indexOf("Swatinem/rust-cache@") > checkout);
    });
}
