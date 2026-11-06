//! The circuit checks a coding agent or a CI job runs on a `.circuit` or `.circuit.layout`:
//! compile, ERC, the design-review suites, and with a board DRC, unrouted connections and
//! whether the board has the schematic's changes; then every `simulation` section, with each
//! measurement checked against its specification limits.
//!
//! The report fails when anything is an error: a compile error, an ERC or review finding of
//! severity `error`, any DRC violation, an unrouted connection, a pending schematic change, a
//! simulation that fails to run, or a measurement outside its limits. Warnings are listed and
//! do not fail it.

use crate::{kind_of, text_of, Files, ToolError};
use runmat_circuit::runtime::{compile, CompileOptions};
use serde::Serialize;

/// One finding of a check.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Finding {
    /// `compile`, `erc`, `review`, `drc`, `unrouted`, `eco` or `simulation`.
    pub check: String,
    pub code: String,
    /// `error`, `warning` or `info`.
    pub severity: String,
    pub message: String,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub items: Vec<String>,
}

/// One measurement of a simulation.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Measurement {
    pub simulation: String,
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub value: Option<f64>,
    pub unit: String,
    /// Inside its limits; `None` when it has none.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pass: Option<bool>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CircuitCheck {
    pub path: String,
    pub passed: bool,
    pub errors: usize,
    pub warnings: usize,
    pub findings: Vec<Finding>,
    pub measurements: Vec<Measurement>,
    /// The checks that ran (`erc`, `review`, `drc`, `simulation:<name>` …).
    pub ran: Vec<String>,
}

/// Runs the checks on `path`; `simulate` also runs its simulation sections.
pub fn check_circuit(path: &str, files: &Files, simulate: bool) -> Result<CircuitCheck, ToolError> {
    let kind = kind_of(path)?;
    if !kind.is_circuit() {
        return Err(ToolError::invalid(format!(
            "the circuit checks run on a .circuit or .circuit.layout; {path} is a {kind:?} file"
        )));
    }
    let text = text_of(files, path)?;
    let out = compile(path, text, files, &CompileOptions::default());
    let mut findings: Vec<Finding> = out
        .diagnostics
        .iter()
        .map(|d| Finding {
            check: "compile".into(),
            code: "compile".into(),
            severity: d.severity.clone(),
            message: format!("{}: {}", d.path, d.message),
            items: vec![],
        })
        .collect();
    let mut ran = vec!["compile".to_string()];
    let mut measurements = Vec::new();
    if let Some(index) = &out.index {
        let rows = |check: &str,
                    rows: &[runmat_circuit::index::CheckRow],
                    always_error: bool|
         -> Vec<Finding> {
            rows.iter()
                .map(|r| Finding {
                    check: check.into(),
                    code: r.code.clone(),
                    severity: if always_error {
                        "error".into()
                    } else {
                        r.severity.clone()
                    },
                    message: r.message.clone(),
                    items: r.items.clone(),
                })
                .collect()
        };
        ran.extend(["erc".to_string(), "review".to_string()]);
        findings.extend(rows("erc", &index.erc, false));
        findings.extend(rows("review", &index.review, false));
        if let Some(board) = &index.board {
            ran.push("drc".into());
            findings.extend(rows("drc", &board.drc, true));
            if !board.unrouted.is_empty() {
                findings.push(Finding {
                    check: "unrouted".into(),
                    code: "unrouted".into(),
                    severity: "error".into(),
                    message: format!("{} connections are not routed", board.unrouted.len()),
                    items: vec![],
                });
            }
            if index.eco.total > 0 {
                findings.push(Finding {
                    check: "eco".into(),
                    code: "board_out_of_sync".into(),
                    severity: "error".into(),
                    message: format!(
                        "the board lacks {} schematic changes (run Update board)",
                        index.eco.total
                    ),
                    items: vec![],
                });
            }
        }
        if simulate {
            for sim in &index.simulations {
                ran.push(format!("simulation:{}", sim.name));
                let circuit = circuit_path(path, files);
                let input = crate::simulate::SimulateInput {
                    path: circuit.clone(),
                    simulation: Some(sim.name.clone()),
                    max_points: Some(2),
                    bins: None,
                    limit_ops: None,
                };
                match crate::simulate::simulate(&input, files) {
                    Ok(report) => {
                        for m in &report.measures {
                            if let Some(e) = &m.error {
                                findings.push(Finding {
                                    check: "simulation".into(),
                                    code: "measure_failed".into(),
                                    severity: "error".into(),
                                    message: format!("{}: {}: {e}", sim.name, m.name),
                                    items: vec![],
                                });
                            }
                            if m.pass == Some(false) {
                                findings.push(Finding {
                                    check: "simulation".into(),
                                    code: "out_of_limits".into(),
                                    severity: "error".into(),
                                    message: format!(
                                        "{}: {} is outside its limits",
                                        sim.name, m.name
                                    ),
                                    items: vec![],
                                });
                            }
                            measurements.push(Measurement {
                                simulation: sim.name.clone(),
                                name: m.name.clone(),
                                value: m.value,
                                unit: m.unit.clone(),
                                pass: m.pass,
                            });
                        }
                    }
                    Err(e) => findings.push(Finding {
                        check: "simulation".into(),
                        code: e.code.clone(),
                        severity: "error".into(),
                        message: format!("{}: {}", sim.name, e.message),
                        items: vec![],
                    }),
                }
            }
        }
    }
    let errors = findings.iter().filter(|f| f.severity == "error").count();
    let warnings = findings.iter().filter(|f| f.severity == "warning").count();
    Ok(CircuitCheck {
        path: path.to_owned(),
        passed: errors == 0,
        errors,
        warnings,
        findings,
        measurements,
        ran,
    })
}

/// The `.circuit` a layout uses (simulations live in the schematic).
fn circuit_path(path: &str, files: &Files) -> String {
    match path.strip_suffix(".layout") {
        Some(c) if files.contains_key(c) => c.to_owned(),
        _ => path.to_owned(),
    }
}
