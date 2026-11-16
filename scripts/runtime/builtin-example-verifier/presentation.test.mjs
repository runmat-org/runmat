import assert from "node:assert/strict";
import test from "node:test";
import { buildReportHtml, buildReportMarkdown } from "./presentation.mjs";

function row() {
    return {
        testCase: {
            description: "A | B", file: "examples/a_b.m", exampleIndex: 0,
            input: "a = `x` \\ y;\r\nb = a < 1;", hasExpectedOutput: true,
            expectedOutput: "a|b\\c", isPlotExample: false, builtin: "plot",
        },
        normalizedActual: "x * y & z", normalizedExpected: "", matches: false,
        imageRelPath: "", imageError: "",
    };
}

test("Markdown report cells preserve literal punctuation and normalized line breaks", () => {
    const record = row();
    const report = buildReportMarkdown([record], [record], "all");
    const cells = report.split("\n").at(-1).split("|");
    assert.equal(cells.length, 7, "five cells and their two table boundaries");
    assert.match(cells[2], /\*\*A &#124; B\*\*/);
    assert.match(cells[2], /examples&#47;a&#95;b&#46;m &#40;example 1&#41;/);
    assert.match(cells[2], /<code>a &#61; &#96;x&#96; &#92; y&#59;<br>b &#61; a &#60; 1&#59;<\/code>/);
    assert.equal(cells[3].trim(), "<code>a&#124;b&#92;c</code>");
    assert.equal(cells[4].trim(), "<code>x &#42; y &#38; z</code>");
    assert.ok(!report.includes("\r"));
});

test("image links use the report's generated PNG namespace in both formats", () => {
    const record = row();
    record.testCase.isPlotExample = true;
    record.imageRelPath = "plot-example-images/0001-plot-ex1.png";
    assert.match(buildReportMarkdown([record], [record], "all"), /\[image\]\(plot-example-images\/0001-plot-ex1\.png\)/);
    assert.match(buildReportHtml([record], [record], "all"), /src="plot-example-images\/0001-plot-ex1\.png"/);
    record.imageRelPath = "other-images/figure.png";
    for (const render of [buildReportMarkdown, buildReportHtml]) {
        assert.throws(() => render([record], [record], "all"), /PNG in plot-example-images/);
    }
    record.imageRelPath = "";
    record.imageError = "width | height\\depth\nnot captured";
    assert.match(buildReportMarkdown([record], [record], "all"), /width &#124; height&#92;depth<br>not captured/);
});

test("empty reports retain their summary without a table", () => {
    assert.equal(buildReportMarkdown([], [], "errors-only"),
        "Results: 0/0 succeeded (0.0%).\nShowing 0 mismatch(es) out of 0 example(s).\n\nNo mismatches detected.\n");
});
