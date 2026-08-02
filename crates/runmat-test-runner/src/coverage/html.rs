use runmat_test::coverage::{CoverageAggregate, CoverageSummary};

use crate::reporter::RenderedReport;

pub(super) fn render(coverage: &CoverageAggregate) -> RenderedReport {
    let mut rows = String::new();
    for file in coverage.files() {
        rows.push_str(&format!(
            "<tr><td><code>{}</code></td><td>{}</td><td>{}</td></tr>",
            html(&file.relative_path),
            cell(file.functions),
            cell(file.statements)
        ));
    }
    let document = format!(
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"><title>RunMat coverage</title><style>{}</style></head><body><main><h1>RunMat coverage</h1><p class=\"revision\">Program revision: <code>{}</code></p><table><thead><tr><th>Source</th><th>Functions</th><th>Statements</th></tr></thead><tbody>{rows}</tbody></table><p class=\"note\">Unsupported regions are excluded from percentages and remain explicit in coverage.json.</p></main></body></html>\n",
        "body{font:15px system-ui,sans-serif;color:#18212f;background:#f7f9fc;margin:0}main{max-width:1000px;margin:3rem auto;padding:0 1.5rem}h1{font-size:2rem}table{width:100%;border-collapse:collapse;background:white;box-shadow:0 1px 4px #0002}th,td{text-align:left;padding:.75rem;border-bottom:1px solid #e3e8ef}th{background:#edf2f7}.revision,.note{color:#5a6678}.pct{font-variant-numeric:tabular-nums}",
        html(coverage.program_revision.as_deref().unwrap_or("none"))
    );
    super::report("coverage.html", "text/html; charset=utf-8", document)
}

fn cell(summary: CoverageSummary) -> String {
    match summary.percentage() {
        Some(percentage) => format!(
            "<span class=\"pct\">{percentage:.1}%</span> ({} / {}, {} unsupported)",
            summary.covered, summary.instrumented, summary.unsupported
        ),
        None => format!("— ({} unsupported)", summary.unsupported),
    }
}

fn html(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;")
}
