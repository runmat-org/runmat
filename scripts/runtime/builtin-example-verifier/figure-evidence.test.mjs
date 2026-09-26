import assert from "node:assert/strict";
import test from "node:test";
import { Script } from "node:vm";
import { createFigureObserver, captureFigureEvidence } from "./figure-evidence.mjs";
import { createRunnerHtml } from "./browser-template.mjs";

const png = new Uint8Array([137, 80, 78, 71, 13, 10, 26, 10, 0]);

test("browser worker embeds standalone evidence helpers with valid syntax", () => {
    const html = createRunnerHtml(1000, 1, 1000, false);
    const encoded = html.match(/const workerSource = (.*);/)[1];
    const source = JSON.parse(encoded);
    assert.doesNotThrow(() => new Script(source));
    assert.match(source, /module\.onFigureEvent\(figures\.observe\)/);
    assert.match(source, /renderedFigureCount/);
});

test("figure lifecycle counts distinct live handles, not updates", () => {
    const observer = createFigureObserver();
    for (const [kind, handle] of [["created", 1], ["updated", 1], ["created", 2], ["closed", 1]]) {
        observer.observe({ kind, handle });
    }
    assert.deepEqual(observer.snapshot(), [2]);
    observer.observe({ kind: "cleared", handle: 2 });
    assert.deepEqual(observer.snapshot(), [2]);
});

test("malformed events fail closed even if the host suppresses callback exceptions", () => {
    const observer = createFigureObserver();
    observer.observe({ kind: "created", handle: NaN });
    assert.throws(() => observer.snapshot(), /Invalid figure/);
});

test("every counted figure must render and present successfully", async () => {
    const presented = [];
    const rendered = [];
    const result = await captureFigureEvidence({
        presentFigureOnSurface(surface, handle) { presented.push([surface, handle]); },
        renderFigureImage(handle) { rendered.push(handle); return png; }
    }, [1, 2], 7);
    assert.equal(result.renderedFigureCount, 2);
    assert.deepEqual(presented, [[7, 1], [7, 2]]);
    assert.deepEqual(rendered, [1, 2]);
});

test("surface errors and invalid image payloads are not success evidence", async () => {
    await assert.rejects(captureFigureEvidence({
        presentFigureOnSurface() { throw new Error("present failed"); },
        renderFigureImage() { return png; }
    }, [1], 7), /present failed/);
    await assert.rejects(captureFigureEvidence({ renderFigureImage() { return [1, 2]; } }, [1], null), /PNG/);
});

test("no observed figures cannot fabricate a rendered count", async () => {
    const result = await captureFigureEvidence({ renderFigureImage() { throw new Error("unexpected"); } }, [], null);
    assert.equal(result.renderedFigureCount, 0);
});
