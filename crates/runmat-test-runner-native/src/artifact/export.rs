use std::collections::BTreeSet;
use std::path::Path;

use runmat_test_runner::reporter::RenderedReport;
use runmat_test_runner::{RunnerError, RunnerResult};

use super::atomic_write::write_atomic;

/// A portable, unambiguous report basename. Built-in report names use only
/// ASCII letters, digits, hyphen, underscore, and dot.
#[derive(Clone, Copy)]
struct ReportFilename<'a>(&'a str);

impl<'a> ReportFilename<'a> {
    fn parse(name: &'a str) -> RunnerResult<Self> {
        let valid_alphabet = name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'));
        let starts_with_letter_or_digit = name
            .as_bytes()
            .first()
            .is_some_and(u8::is_ascii_alphanumeric);
        let ends_with_letter_or_digit = name
            .as_bytes()
            .last()
            .is_some_and(u8::is_ascii_alphanumeric);
        let stem = name.split('.').next().unwrap_or_default();
        let reserved = matches!(
            stem.to_ascii_uppercase().as_str(),
            "CON"
                | "PRN"
                | "AUX"
                | "NUL"
                | "COM1"
                | "COM2"
                | "COM3"
                | "COM4"
                | "COM5"
                | "COM6"
                | "COM7"
                | "COM8"
                | "COM9"
                | "LPT1"
                | "LPT2"
                | "LPT3"
                | "LPT4"
                | "LPT5"
                | "LPT6"
                | "LPT7"
                | "LPT8"
                | "LPT9"
        );
        if !valid_alphabet || !starts_with_letter_or_digit || !ends_with_letter_or_digit || reserved
        {
            return Err(RunnerError::Artifact(format!(
                "report export requires a portable flat filename: '{name}'"
            )));
        }
        Ok(Self(name))
    }
}

/// Publish the requested reports as stable files directly in `root`.
///
/// The run-scoped artifact store remains separate: export may replace only the
/// validated names supplied in this call and never owns or cleans the directory.
pub async fn export_reports(root: &Path, reports: &[RenderedReport]) -> RunnerResult<()> {
    let mut names = BTreeSet::new();
    let mut validated = Vec::with_capacity(reports.len());
    for report in reports {
        let name = ReportFilename::parse(&report.name)?;
        if !names.insert(name.0.to_ascii_lowercase()) {
            return Err(RunnerError::Artifact(format!(
                "duplicate report export filename: '{}'",
                name.0
            )));
        }
        validated.push(name);
    }

    for (report, name) in reports.iter().zip(validated) {
        write_atomic(&root.join(name.0), &report.bytes)
            .await
            .map_err(|error| RunnerError::Artifact(error.to_string()))?;
    }
    Ok(())
}
