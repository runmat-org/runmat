// @ts-check

import { writeFileSync } from "node:fs";
import { join } from "node:path";

export function resolveReportMode() {
    const args = new Set(process.argv.slice(2));
    if (args.has("--errors-only")) {
        return "errors-only";
    }
    if (args.has("--all")) {
        return "all";
    }
    const raw = process.env.RUNMAT_EXAMPLE_REPORT_MODE;
    if (!raw) {
        return "all";
    }
    const normalized = raw.trim().toLowerCase();
    if (normalized === "errors-only" || normalized === "errors" || normalized === "mismatches") {
        return "errors-only";
    }
    return "all";
}

/**
 * @param {{ testCase: ExampleCase, normalizedExpected: string, normalizedActual: string, matches: boolean }[]} rows
 * @param {"all" | "errors-only"} reportMode
 */
export function filterReportRows(rows, reportMode) {
    if (reportMode === "errors-only") {
        return rows.filter((row) => !row.matches);
    }
    return rows;
}

/**
 * @param {string} text
 */
export function normalizeOutput(text) {
    if (typeof text !== "string") {
        return "";
    }
    const lines = text.replace(/\r\n/g, "\n").split("\n");
    const stripped = lines.map((line) =>
        line.replace(/^\s*Columns?\s+\d+\s+through\s+\d+\s*$/i, "")
            .replace(/^\s*Column\s+\d+\s*$/i, "")
            .replace(/^\s*\d+(?:\s*[x×]\s*\d+)+\s+\w+(?:\s+\w+)*(?:\s+array)?\s*$/i, "")
            .replace(/^\s*\d+(?:\s*[x×]\s*\d+)+\s*$/i, "")
            .replace(/^\s*(?:[A-Za-z_]\w*)?\(\s*:\s*,\s*:\s*(?:,\s*\d+\s*)+\)\s*=\s*$/i, "")
            .replace(/^\s*[A-Za-z_]\w*(?:\([^)]*\))?\s*=\s*/, "")
    );
    const joined = stripped.join(" ");
    const separatedAssignments = joined.replace(/(\d)([A-Za-z_]\w*\s*=\s*)/g, "$1 $2");
    const withoutAssignments = separatedAssignments.replace(
        /\b[A-Za-z_]\w*(?:\([^)]*\))?\s*=\s*/g,
        ""
    );
    const withoutInlineDims = withoutAssignments.replace(
        /(^|\s)\d+\s*[x×]\s*\d+(?:\s*[x×]\s*\d+)*\s+(?=(?:[-+]?(?:\d|\.\d)|NaN|Inf|-Inf))/gi,
        "$1"
    );
    const withoutHeaders = withoutInlineDims.replace(
        /\b\d+\s*[x×]\s*\d+(?:\s*[x×]\s*\d+)*\s+(?:gpuArray\s*)?(?:sparse\s+)?(?:complex\s+)?(?:logical\s+)?(?:logical|double|single|char|string|cell|struct|table|categorical|datetime|duration)(?:\s+array)?\b/gi,
        " "
    );
    const withoutBrackets = withoutHeaders.replace(/[\[\]{};,]/g, " ");
    const withoutQuotes = withoutBrackets.replace(/["']/g, " ");
    const normalizedBooleans = withoutQuotes
        .replace(/\btrue\b/gi, "1")
        .replace(/\bfalse\b/gi, "0")
        .replace(/\blogical\((0|1)\)\b/gi, "$1");
    const normalizedConstants = normalizedBooleans
        .replace(/\bpi\b/gi, String(Math.PI));
    const strippedMetadata = normalizedConstants
        .replace(/(?:GpuTensor|Tensor|ComplexTensor)\(\s*(?:shape\s*=\s*)?[^)]*\)/g, " ")
        .replace(/\b\d+(?:\s*[x×]\s*\d+)+\s*(?:gpuArray\s*)?(?:logical|double|single|char|string|cell)?\s*array\b/gi, " ")
        .replace(/\b(?:gpuArray|logical|double|single|string|char|cell|Tensor|ComplexTensor|GpuTensor)\b/gi, " ");
    const normalizedComplex = strippedMetadata
        .replace(/(\d+\.\d+)(\d+\.\d+[ij])/g, "$1+$2")
        .replace(/(\d)\s*([+-])\s*(?=(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?[ij]\b)/g, "$1$2")
        .replace(/\+\s*-/g, "-")
        .replace(/-\s*\+/g, "-")
        .replace(/\+\s*\+/g, "+");
    const normalizedNumbers = normalizedComplex.replace(
        /[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?/g,
        (match) => {
            if (!/[.eE]/.test(match)) {
                return match;
            }
            const value = Number(match);
            if (!Number.isFinite(value)) {
                return match;
            }
            let formatted = value.toFixed(4);
            formatted = formatted.replace(/\.?0+$/, "");
            if (formatted === "-0") {
                formatted = "0";
            }
            // Preserve leading + sign for positive numbers (important for complex imaginary parts)
            if (match.startsWith('+') && !formatted.startsWith('-')) {
                formatted = '+' + formatted;
            }
            return formatted;
        }
    );
    const withoutZeroPlus = normalizedNumbers.replace(/\b0\s*\+\s*/g, "");
    const withoutZeroMinus = withoutZeroPlus.replace(/\b0-(?=\d+(?:\.\d+)?[ij]\b)/g, "-");
    const withoutImaginaryZero = withoutZeroMinus
        .replace(/\s*[+-]\s*0[ij]\b/g, "")
        .replace(/\b0[ij]\b/g, "0");
    const compact = withoutImaginaryZero.trim().replace(/\s+/g, " ");
    const fixedZeroConcat = compact
        .replace(/(?<![0-9.])0(?=\d+(?:\.\d+)?[ij]\b)/g, "")
        .replace(/(?<![0-9.])0(?=0[ij]\b)/g, "")
        .replace(/(\d+\.\d+)0[ij]\b/g, "$1")
        .replace(/\b0[ij]\b/g, "0")
        .replace(/\s+/g, " ")
        .trim();
    return fixedZeroConcat;
}

/**
 * @param {RunnerResult | undefined} result
 */
export function formatExecutionOutput(result) {
    if (!result) {
        return "";
    }
    if (result.errorText) {
        return result.errorText;
    }
    if (result.stdoutText && result.valueText) {
        const normalizedStdout = normalizeOutput(result.stdoutText);
        const normalizedValue = normalizeOutput(result.valueText);
        if (!normalizedValue) {
            return result.stdoutText;
        }
        if (normalizedStdout && normalizedStdout === normalizedValue) {
            return result.valueText;
        }
        if (normalizedStdout) {
            const suffixIndex = normalizedStdout.lastIndexOf(normalizedValue);
            const isSuffix = suffixIndex >= 0
                && suffixIndex + normalizedValue.length === normalizedStdout.length
                && (suffixIndex === 0 || normalizedStdout[suffixIndex - 1] === " ");
            if (isSuffix) {
                return result.stdoutText;
            }
        }
        return `${result.stdoutText}\n${result.valueText}`;
    }
    return result.stdoutText || result.valueText || "";
}

/**
 * @param {ExampleCase} testCase
 */
/**
 * @param {{ testCase: ExampleCase, normalizedExpected: string, normalizedActual: string, imageRelPath: string, imageError: string, matches: boolean }[]} rows
 */
export function buildReportHtml(allRows, reportRows, reportMode) {
    const title = "RunMat Builtins Example Output Report";
    const totalCount = allRows.length;
    const errorCount = allRows.filter((row) => !row.matches).length;
    const successCount = totalCount - errorCount;
    const successPercent = totalCount === 0 ? "0.0" : ((successCount / totalCount) * 100).toFixed(1);
    const renderedRows = reportRows.map((row) => {
        const statusClass = row.matches ? "status-ok" : "status-bad";
        const statusSymbol = row.matches ? "&#10003;" : "X";
        const statusLabel = row.matches ? "Match" : "Mismatch";
        const description = escapeHtml(row.testCase.description);
        const sourceInfo = escapeHtml(
            `${row.testCase.file} (example ${row.testCase.exampleIndex + 1})`
        );
        const input = escapeHtml(row.testCase.input ?? "");
        const expected = escapeHtml(row.testCase.hasExpectedOutput ? row.testCase.expectedOutput : "(execution-only: no expected output)");
        const actual = escapeHtml(row.normalizedActual);
        const imageCell = row.testCase.isPlotExample
            ? row.imageRelPath
                ? `<a href="${escapeHtml(row.imageRelPath)}" target="_blank" rel="noopener"><img src="${escapeHtml(row.imageRelPath)}" alt="${escapeHtml(row.testCase.builtin)} plot image" class="plot-preview" /></a>`
                : `<div class="meta">No image captured</div>${row.imageError ? `<pre>${escapeHtml(row.imageError)}</pre>` : ""}`
            : `<div class="meta">N/A</div>`;
        return `<tr>
      <td>
        <div class="status ${statusClass}" title="${statusLabel}">${statusSymbol}</div>
        <div class="meta">${description}</div>
        <div class="meta">${sourceInfo}</div>
        <pre>${input}</pre>
      </td>
      <td><pre>${expected}</pre></td>
      <td><pre>${actual}</pre></td>
      <td>${imageCell}</td>
    </tr>`;
    }).join("");

    return `<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>${escapeHtml(title)}</title>
    <style>
      :root {
        color-scheme: light;
      }
      body {
        margin: 24px;
        font-family: "Helvetica Neue", Arial, sans-serif;
        color: #111;
      }
      .meta {
        color: #555;
        font-size: 13px;
      }
      table {
        border-collapse: collapse;
        width: 100%;
        table-layout: fixed;
      }
      th, td {
        border: 1px solid #ddd;
        padding: 12px;
        vertical-align: top;
      }
      th {
        text-align: left;
        background: #f6f7f9;
        font-weight: 600;
        position: sticky;
        top: 0;
        z-index: 1;
      }
      td {
        background: #fff;
      }
      .status {
        font-size: 16px;
        font-weight: 700;
        margin-bottom: 6px;
      }
      .status-ok {
        color: #0a7c3a;
      }
      .status-bad {
        color: #b42318;
      }
      pre {
        margin: 0;
        white-space: pre-wrap;
        word-break: break-word;
        font-family: "SFMono-Regular", Menlo, Consolas, "Liberation Mono", monospace;
        font-size: 13px;
      }
      .plot-preview {
        width: 100%;
        max-width: 360px;
        border: 1px solid #ddd;
        border-radius: 6px;
        background: #fff;
      }
    </style>
  </head>
  <body>
    <h1>${escapeHtml(title)}</h1>
    <p class="meta">Results: ${successCount}/${totalCount} succeeded (${successPercent}%).</p>
    <p class="meta">${
        reportMode === "errors-only"
            ? `Showing ${reportRows.length} mismatch(es) out of ${totalCount} example(s).`
            : `Showing all ${reportRows.length} example(s).`
    }</p>
    ${
        reportRows.length === 0
            ? "<p class=\"meta\">No mismatches detected.</p>"
            : `<table>
      <thead>
        <tr>
          <th>Input</th>
          <th>Normalized JSON Output</th>
          <th>Normalized WASM Output</th>
          <th>Plot Image</th>
        </tr>
      </thead>
      <tbody>
        ${renderedRows}
      </tbody>
    </table>`
    }
  </body>
</html>`;
}

/**
 * @param {{ testCase: ExampleCase, normalizedExpected: string, normalizedActual: string, imageRelPath: string, imageError: string, matches: boolean }[]} allRows
 * @param {{ testCase: ExampleCase, normalizedExpected: string, normalizedActual: string, imageRelPath: string, imageError: string, matches: boolean }[]} reportRows
 * @param {"all" | "errors-only"} reportMode
 */
export function buildReportMarkdown(allRows, reportRows, reportMode) {
    const totalCount = allRows.length;
    const errorCount = allRows.filter((row) => !row.matches).length;
    const successCount = totalCount - errorCount;
    const successPercent = totalCount === 0 ? "0.0" : ((successCount / totalCount) * 100).toFixed(1);
    const summary = reportMode === "errors-only"
        ? `Showing ${reportRows.length} mismatch(es) out of ${totalCount} example(s).`
        : `Showing all ${reportRows.length} example(s).`;
    const tableHeader = "| Status | Description / Input | Expected | Actual | Plot Image |\n| :--- | :--- | :--- | :--- | :--- |\n";
    const tableRows = reportRows.map((row) => {
        const status = row.matches ? "✅" : "❌";
        const description = row.testCase.description;
        const sourceInfo = `${row.testCase.file} (example ${row.testCase.exampleIndex + 1})`;
        // Escape pipe characters in markdown cells and use <br> for newlines in table cells
        const input = (row.testCase.input ?? "").replace(/\|/g, "\\|").replace(/\n/g, "<br>");
        const expectedRaw = row.testCase.hasExpectedOutput ? row.testCase.expectedOutput : "(execution-only: no expected output)";
        const expected = expectedRaw.replace(/\|/g, "\\|").replace(/\n/g, "<br>");
        const actual = row.normalizedActual.replace(/\|/g, "\\|").replace(/\n/g, "<br>");
        const imageCell = row.testCase.isPlotExample
            ? row.imageRelPath
                ? `[image](${row.imageRelPath.replace(/\|/g, "\\|")})`
                : row.imageError
                    ? `(no image captured: ${row.imageError.replace(/\|/g, "\\|")})`
                    : "(no image captured)"
            : "N/A";

        return `| ${status} | **${description}**<br>${sourceInfo}<br>\`${input}\` | \`${expected}\` | \`${actual}\` | ${imageCell} |`;
    }).join("\n");

    const header = `Results: ${successCount}/${totalCount} succeeded (${successPercent}%).\n${summary}\n\n`;
    if (!tableRows) {
        return `${header}No mismatches detected.\n`;
    }
    return header + tableHeader + tableRows;
}

/**
 * @param {string} value
 */
function escapeHtml(value) {
    return String(value)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#39;");
}

/**
 * @param {string} outDir
 * @param {ExampleCase} testCase
 * @param {RunnerResult | undefined} result
 */
export function writePlotImageArtifact(outDir, testCase, result) {
    if (!testCase.isPlotExample || !result || typeof result.figurePngBase64 !== "string" || result.figurePngBase64.length === 0) {
        return "";
    }
    try {
        const stem = `${String(testCase.id).padStart(4, "0")}-${sanitizeFileStem(testCase.builtin)}-ex${testCase.exampleIndex + 1}`;
        const fileName = `${stem}.png`;
        const absPath = join(outDir, fileName);
        writeFileSync(absPath, Buffer.from(result.figurePngBase64, "base64"));
        return join("plot-example-images", fileName).replace(/\\/g, "/");
    } catch (_err) {
        return "";
    }
}

/**
 * @param {string} value
 */
function sanitizeFileStem(value) {
    return String(value)
        .toLowerCase()
        .replace(/[^a-z0-9]+/g, "-")
        .replace(/^-+|-+$/g, "")
        .slice(0, 60) || "plot";
}

/**
 * @param {{ repoRoot: string, chromeWrapper: string, runnerHtml: string, casesJson: string, overallTimeoutMs?: number, wasmModule: string, wasmBinary: string }} options
 * @returns {Promise<RunnerResult[]>}
 */
