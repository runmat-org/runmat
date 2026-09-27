import assert from "node:assert/strict";
import test from "node:test";
import { Script } from "node:vm";
import { createFigureObserver, captureFigureEvidence, withFigureSurface } from "./figure-evidence.mjs";
import { createRunnerHtml } from "./browser-template.mjs";

const png = new Uint8Array([137, 80, 78, 71, 13, 10, 26, 10, 0]);

function surfaceFixture() {
    const events = [];
    return {
        events,
        Canvas: class { constructor() { events.push("canvas"); } },
        session: { dispose() { events.push("dispose"); } },
        module: {
            createPlotSurface() { events.push("create"); return 7; },
            resizePlotSurface() { events.push("resize"); },
            presentFigureOnSurface() {},
            destroyPlotSurface(id) { events.push(`destroy:${id}`); }
        }
    };
}

test("required surface rejects missing APIs and canvas, disposing the session", async () => {
    for (const missing of ["createPlotSurface", "resizePlotSurface", "presentFigureOnSurface", "destroyPlotSurface", "Canvas"]) {
        const f = surfaceFixture();
        if (missing === "Canvas") f.Canvas = undefined;
        else delete f.module[missing];
        await assert.rejects(withFigureSurface(f.module, f.session, true, f.Canvas, () => assert.fail("must not execute")), /unavailable/);
        assert.deepEqual(f.events, ["dispose"]);
    }
});

test("creation failure disposes without executing or destroying an unacquired surface", async () => {
    const f = surfaceFixture();
    f.module.createPlotSurface = () => { throw new Error("create failed"); };
    await assert.rejects(withFigureSurface(f.module, f.session, true, f.Canvas, () => assert.fail("must not execute")), /create failed/);
    assert.deepEqual(f.events, ["canvas", "dispose"]);
});

test("resize failure retains the acquired surface for cleanup and disposes", async () => {
    const f = surfaceFixture();
    f.module.resizePlotSurface = () => { throw new Error("resize failed"); };
    await assert.rejects(withFigureSurface(f.module, f.session, true, f.Canvas, () => assert.fail("must not execute")), /resize failed/);
    assert.deepEqual(f.events, ["canvas", "create", "destroy:7", "dispose"]);
});

test("surface scope disposes after normal execution, body failure, and cleanup failure", async () => {
    for (const failure of [null, "body", "destroy"]) {
        const f = surfaceFixture();
        if (failure === "destroy") f.module.destroyPlotSurface = () => { f.events.push("destroy:7"); throw new Error("destroy failed"); };
        const execution = withFigureSurface(f.module, f.session, true, f.Canvas, (id) => {
            assert.equal(id, 7);
            f.events.push("body");
            if (failure === "body") throw new Error("body failed");
            return 42;
        });
        if (failure) await assert.rejects(execution, new RegExp(`${failure} failed`));
        else assert.equal(await execution, 42);
        assert.deepEqual(f.events, ["canvas", "create", "resize", "body", "destroy:7", "dispose"]);
    }
});

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
