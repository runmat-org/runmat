//! Circuit files in `runmat check` and `runmat test`: each `.circuit` or `.circuit.layout` runs
//! the design tools' circuit checks (compile, ERC, the design-review suites; with a board DRC,
//! unrouted connections and pending schematic changes; every simulation section's measurements
//! against their limits). An error anywhere fails the file; warnings are listed.

use anyhow::{Context, Result};
use runmat_design_tools::ci::{check_circuit, report_lines, CircuitCheck};
use runmat_design_tools::inspect::m_values::with_m_values;
use runmat_design_tools::workspace::{is_circuit_file, read_workspace, source_path};
use std::path::{Path, PathBuf};

pub fn is_circuit(path: &Path) -> bool {
    is_circuit_file(path)
}

/// Checks one circuit file. The workspace is the file's folder and everything under it, so the
/// layout, the lock and the model libraries the design includes are read.
pub fn check_file(file: &Path, simulate: bool) -> Result<(CircuitCheck, PathBuf)> {
    let file = std::fs::canonicalize(file)
        .with_context(|| format!("failed to locate {}", file.display()))?;
    let base = file.parent().unwrap_or(Path::new("/")).to_path_buf();
    let files = read_workspace(&base);
    // Values computed by `.m` functions run in a RunMat session (PCB-38).
    let mut report = with_m_values(&files, || {
        check_circuit(&source_path(&file), &files, simulate)
    })
    .map_err(|e| anyhow::anyhow!("{}: {e}", file.display()))?;
    report.path = file.to_string_lossy().into_owned();
    Ok((report, base))
}

/// Prints the reports (text lines, or one JSON array) and says whether every file passed.
pub fn print_reports(reports: &[(CircuitCheck, PathBuf)], json: bool) -> Result<bool> {
    if json {
        let all: Vec<&CircuitCheck> = reports.iter().map(|(r, _)| r).collect();
        println!("{}", serde_json::to_string_pretty(&all)?);
    } else {
        for (report, base) in reports {
            for line in report_lines(report, &base.to_string_lossy()) {
                println!("{line}");
            }
        }
    }
    Ok(reports.iter().all(|(r, _)| r.passed))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The drone's flight controller: the checks run ERC, the review suites and DRC through
    /// `runmat check` and `runmat test`, and the CLI parses `--no-simulate`.
    #[test]
    fn checks_the_flight_controller() {
        let file = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../runmat-cad/wasm/tests/drone/electrical/flight_controller/flight_controller.circuit.layout");
        assert!(is_circuit(&file));
        assert!(!is_circuit(Path::new("model.fea")));
        let (report, base) = check_file(&file, false).expect("checks run");
        for check in ["erc", "review", "drc"] {
            assert!(
                report.ran.iter().any(|r| r == check),
                "{check} ran: {:?}",
                report.ran
            );
        }
        let lines = report_lines(&report, &base.to_string_lossy());
        assert!(lines
            .last()
            .unwrap()
            .starts_with("flight_controller.circuit.layout: "));
        let mut windows_report = report.clone();
        windows_report.path = r"C:\work\flight_controller.circuit.layout".into();
        assert_eq!(report_lines(&windows_report, r"C:\work"), lines);
        use clap::Parser;
        let cli = crate::cli::Cli::try_parse_from([
            "runmat",
            "check",
            "board.circuit.layout",
            "--no-simulate",
        ])
        .unwrap();
        assert!(matches!(
            cli.command,
            Some(crate::cli::Commands::Check {
                no_simulate: true,
                ..
            })
        ));
    }
}
