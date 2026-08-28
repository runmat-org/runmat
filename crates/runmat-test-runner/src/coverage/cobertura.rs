use runmat_test::coverage::{CoverageAggregate, CoverageMetric};
use std::collections::BTreeMap;

use crate::reporter::RenderedReport;

pub(super) fn render(coverage: &CoverageAggregate) -> RenderedReport {
    let functions = coverage.summary(CoverageMetric::Function);
    let all_lines = coverage
        .files()
        .into_iter()
        .flat_map(|file| lines_for_file(coverage, &file.owner_identity, &file.relative_path))
        .collect::<Vec<_>>();
    let lines_valid = all_lines.len() as u64;
    let lines_covered = all_lines.iter().filter(|(_, hits)| *hits != 0).count() as u64;
    let line_rate = rate(lines_covered, lines_valid);
    let branch_rate = 0.0;
    let mut output = format!(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<coverage version=\"runmat\" lines-valid=\"{}\" lines-covered=\"{}\" line-rate=\"{line_rate:.6}\" branches-valid=\"0\" branches-covered=\"0\" branch-rate=\"{branch_rate:.6}\">\n  <sources/>\n  <packages>\n",
        lines_valid, lines_covered
    );
    for file in coverage.files() {
        let lines = lines_for_file(coverage, &file.owner_identity, &file.relative_path);
        let file_rate = rate(
            lines.iter().filter(|(_, hits)| **hits != 0).count() as u64,
            lines.len() as u64,
        );
        output.push_str(&format!(
            "    <package name=\"{}\" line-rate=\"{file_rate:.6}\" branch-rate=\"0\">\n      <classes>\n        <class name=\"{}\" filename=\"{}\" line-rate=\"{file_rate:.6}\" branch-rate=\"0\">\n          <methods/>\n          <lines>\n",
            xml(&file.owner_identity),
            xml(&file.relative_path),
            xml(&file.relative_path),
        ));
        for (line, hits) in lines {
            output.push_str(&format!(
                "            <line number=\"{}\" hits=\"{}\" branch=\"false\"/>\n",
                line, hits
            ));
        }
        output.push_str("          </lines>\n        </class>\n      </classes>\n    </package>\n");
    }
    output.push_str(&format!(
        "  </packages>\n  <!-- functions-covered={} functions-valid={} -->\n</coverage>\n",
        functions.covered, functions.instrumented
    ));
    super::report("coverage.xml", "application/xml", output)
}

fn lines_for_file(
    coverage: &CoverageAggregate,
    owner_identity: &str,
    relative_path: &str,
) -> BTreeMap<u32, u64> {
    let mut lines = BTreeMap::<u32, u64>::new();
    for site in coverage.sites.iter().filter(|site| {
        site.instrumented
            && site.metric == CoverageMetric::Statement
            && site.owner_identity == owner_identity
            && site.relative_path == relative_path
    }) {
        let hits = lines.entry(site.start_line).or_default();
        *hits = hits.saturating_add(coverage.count(site));
    }
    lines
}

fn rate(covered: u64, valid: u64) -> f64 {
    if valid == 0 {
        0.0
    } else {
        covered as f64 / valid as f64
    }
}

fn xml(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}
