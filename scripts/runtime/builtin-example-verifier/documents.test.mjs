import assert from "node:assert/strict";
import test from "node:test";
import { collectCases, matchesVerification } from "./documents.mjs";

function collect(verification, category = "general") {
    return collectCases([{
        key: "example_identity",
        title: "Example identity",
        authority: "catalog",
        category,
        examples: [{
            id: "semantic-oracle",
            description: "Exercise the declared semantic oracle",
            input: "x = 1;\n",
            harness: "BrowserGraphics",
            verification
        }]
    }])[0];
}

test("Figure assertions execute even outside a plotting taxonomy", () => {
    const result = collect({ Figure: { minimum_figures: 2, assertions: "assert(x == 2);" } });
    assert.equal(result.input, "x = 1;\nassert(x == 2);");
    assert.equal(result.isPlotExample, true);
    assert.equal(result.verification.Figure.minimum_figures, 2);
});

test("ordinary assertions retain their execution behavior", () => {
    const result = collect({ Assertions: { source: "assert(x == 1);" } });
    assert.equal(result.input, "x = 1;\nassert(x == 1);");
    assert.equal(result.isPlotExample, false);
});

test("empty Figure assertions preserve the original program", () => {
    const result = collect({ Figure: { minimum_figures: 1, assertions: "" } });
    assert.equal(result.input, "x = 1;\n");
    assert.equal(result.isPlotExample, true);
});

test("expected errors are not rewritten as assertion code", () => {
    const result = collect({ ExpectedError: { identifier: "RunMat:test:Expected" } });
    assert.equal(result.input, "x = 1;\n");
});

test("Figure oracle requires its minimum distinct rendered count", () => {
    const example = collect({ Figure: { minimum_figures: 2, assertions: "" } });
    const matches = (result, image = "figure.png") => matchesVerification(example, result, "", "", false, image);
    assert.equal(matches({ renderedFigureCount: 1 }), false);
    assert.equal(matches({ renderedFigureCount: 2 }), true);
    assert.equal(matches({ figureVerified: true }), false);
    assert.equal(matches({ renderedFigureCount: 2, figureImageError: "failed" }), false);
    assert.equal(matches({ renderedFigureCount: 2 }, ""), false);
    assert.equal(matches({ renderedFigureCount: 2, figureVerified: true }, ""), true);
});
