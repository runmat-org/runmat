mod cobertura;
mod html;
mod json;
mod lcov;
mod view;

use runmat_test::coverage::{CoverageAggregate, CoverageFilter};

use crate::reporter::RenderedReport;
use crate::{RunnerError, RunnerResult};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CoverageReportFormat {
    Json,
    Lcov,
    Cobertura,
    Html,
}

pub fn render_coverage_reports(
    coverage: &CoverageAggregate,
    filter: &CoverageFilter,
    formats: &[CoverageReportFormat],
) -> RunnerResult<Vec<RenderedReport>> {
    let coverage = view::filtered(coverage, filter);
    let mut rendered = Vec::new();
    for format in formats {
        let report = match format {
            CoverageReportFormat::Json => json::render(&coverage),
            CoverageReportFormat::Lcov => Ok(lcov::render(&coverage)),
            CoverageReportFormat::Cobertura => Ok(cobertura::render(&coverage)),
            CoverageReportFormat::Html => Ok(html::render(&coverage)),
        }?;
        if !rendered
            .iter()
            .any(|existing: &RenderedReport| existing.name == report.name)
        {
            rendered.push(report);
        }
    }
    Ok(rendered)
}

fn report(name: &str, media_type: &str, text: String) -> RenderedReport {
    RenderedReport {
        name: name.into(),
        media_type: media_type.into(),
        bytes: text.into_bytes(),
    }
}

fn serialization_error(error: impl std::fmt::Display) -> RunnerError {
    RunnerError::Reporter(format!("coverage serialization failed: {error}"))
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use runmat_test::coverage::{CoverageMetric, CoverageSite};

    use super::*;

    #[test]
    fn all_formats_are_deterministic_and_machine_readable() {
        let site = CoverageSite {
            id: "site".into(),
            counter_key: 1,
            metric: CoverageMetric::Statement,
            owner_identity: "root".into(),
            relative_path: "src/example.m".into(),
            semantic_path: "example".into(),
            source_id: 0,
            start_byte: 0,
            end_byte: 5,
            start_line: 1,
            start_column: 1,
            end_line: 1,
            end_column: 6,
            instrumented: true,
            unsupported_reason: None,
        };
        let mut same_line = site.clone();
        same_line.id = "same-line".into();
        same_line.counter_key = 2;
        same_line.start_column = 7;
        same_line.end_column = 10;
        let coverage = CoverageAggregate {
            program_revision: Some("program".into()),
            sites: vec![site, same_line],
            counts: BTreeMap::from([("site".into(), 2), ("same-line".into(), 0)]),
        };
        let formats = [
            CoverageReportFormat::Json,
            CoverageReportFormat::Lcov,
            CoverageReportFormat::Cobertura,
            CoverageReportFormat::Html,
            CoverageReportFormat::Json,
        ];
        let first =
            render_coverage_reports(&coverage, &CoverageFilter::default(), &formats).unwrap();
        let second =
            render_coverage_reports(&coverage, &CoverageFilter::default(), &formats).unwrap();
        assert_eq!(first, second);
        assert_eq!(
            first
                .iter()
                .map(|report| report.name.as_str())
                .collect::<Vec<_>>(),
            vec![
                "coverage.json",
                "coverage.lcov",
                "coverage.xml",
                "coverage.html"
            ]
        );
        serde_json::from_slice::<serde_json::Value>(&first[0].bytes).unwrap();
        assert!(String::from_utf8_lossy(&first[1].bytes).contains("DA:1,2"));
        let cobertura = String::from_utf8_lossy(&first[2].bytes);
        assert!(cobertura.contains("<coverage"));
        assert!(cobertura.contains("lines-valid=\"1\""));
        assert!(cobertura.contains("lines-covered=\"1\""));
        assert!(String::from_utf8_lossy(&first[3].bytes).contains("<!doctype html>"));
    }
}
