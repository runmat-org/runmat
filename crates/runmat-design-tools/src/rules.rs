//! `design.drc` and `design.erc`: rule violations with what a client needs to fix them.

use crate::check::DesignDiagnostic;
use crate::{kind_of, text_of, Files, Locator, Range, ToolError};
use runmat_circuit::runtime::{rule_check, Violation as CircuitViolation};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct RulesInput {
    /// Workspace path of the `.circuit` or `.circuit.layout`.
    pub path: String,
    /// Only violations of these severities (`error`, `warning`, `info`); all when absent.
    #[serde(default)]
    pub severities: Option<Vec<String>>,
    /// Only violations whose objects match (designator, `U3.4`, net name; `*` globs).
    #[serde(default)]
    pub filter: Option<String>,
    /// At most this many violations (default 500).
    #[serde(default)]
    pub limit: Option<usize>,
}

/// One violation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct Violation {
    pub code: String,
    /// `error`, `warning` or `info`.
    pub severity: String,
    pub message: String,
    /// The rule that set the limit (`rule power-width`, `clearance`, `connectivity`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rule: Option<String>,
    /// The objects involved: designators, `U3.4` pins and pads, nets, `track 12`, `via 3`,
    /// `pour GND`, `board edge`.
    pub objects: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub layer: Option<String>,
    /// Board location in millimetres (DRC).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub location_mm: Option<[f64; 2]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub actual: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub required: Option<f64>,
    /// `mm` or `count`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub unit: Option<String>,
    /// The rule or declaration in the source.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub range: Option<Range>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct RulesOutput {
    pub path: String,
    /// `drc` or `erc`.
    pub check: String,
    /// The check ran (the files compile, and DRC has a board).
    pub ran: bool,
    pub errors: usize,
    pub warnings: usize,
    /// Violations before `limit` and `filter`.
    pub total: usize,
    pub violations: Vec<Violation>,
    /// Why the check didn't run: the compile errors.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub diagnostics: Vec<DesignDiagnostic>,
    pub summary: String,
}

pub fn drc(input: &RulesInput, files: &Files) -> Result<RulesOutput, ToolError> {
    run(input, files, "drc")
}

pub fn erc(input: &RulesInput, files: &Files) -> Result<RulesOutput, ToolError> {
    run(input, files, "erc")
}

fn convert(v: &CircuitViolation) -> Violation {
    Violation {
        code: v.code.clone(),
        severity: v.severity.clone(),
        message: v.message.clone(),
        rule: v.rule.clone(),
        objects: v.items.clone(),
        layer: v.layer.clone(),
        location_mm: v.at_mm,
        actual: v.actual,
        required: v.required,
        unit: v.unit.clone(),
        range: v.source.as_ref().map(|s| Range {
            path: s.path.clone(),
            start: s.start,
            end: s.end,
            line: s.line,
            column: s.column,
        }),
    }
}

fn run(input: &RulesInput, files: &Files, check: &str) -> Result<RulesOutput, ToolError> {
    let kind = kind_of(&input.path)?;
    if !kind.is_circuit() {
        return Err(ToolError::invalid(format!(
            "{} runs on a .circuit or .circuit.layout; {} is a {kind:?} file",
            if check == "drc" { "DRC" } else { "ERC" },
            input.path
        )));
    }
    let text = text_of(files, &input.path)?;
    let result = rule_check(&input.path, text, files, &Default::default());
    let diagnostics: Vec<DesignDiagnostic> = result
        .diagnostics
        .iter()
        .map(|d| {
            let t = files.get(&d.path).map(String::as_str).unwrap_or(text);
            DesignDiagnostic {
                id: None,
                severity: d.severity.clone(),
                code: None,
                message: d.message.clone(),
                cause: "root".into(),
                root: None,
                subject: None,
                failed_features: Vec::new(),
                range: Some(Locator::new(t).range_utf16(&d.path, d.start, d.end)),
                repairs: Vec::new(),
            }
        })
        .collect();
    let (ran, all): (bool, Vec<Violation>) = if check == "drc" {
        match &result.drc {
            Some(v) => (true, v.iter().map(convert).collect()),
            None => (false, Vec::new()),
        }
    } else {
        (result.compiled, result.erc.iter().map(convert).collect())
    };
    let errors = all.iter().filter(|v| v.severity == "error").count();
    let warnings = all.iter().filter(|v| v.severity == "warning").count();
    let total = all.len();
    let violations: Vec<Violation> = all
        .into_iter()
        .filter(|v| {
            input
                .severities
                .as_ref()
                .is_none_or(|s| s.iter().any(|x| x == &v.severity))
        })
        .filter(|v| {
            input.filter.is_none()
                || v.objects
                    .iter()
                    .any(|o| crate::name_matches(input.filter.as_deref(), o))
        })
        .take(input.limit.unwrap_or(500))
        .collect();
    let summary = if !ran && check == "drc" && result.compiled {
        format!(
            "{}: no board to check (the circuit has no .circuit.layout)",
            input.path
        )
    } else if !ran {
        format!(
            "{}: the files don't compile, so {} didn't run",
            input.path,
            check.to_uppercase()
        )
    } else {
        format!(
            "{}: {} {} error(s), {} warning(s)",
            input.path,
            check.to_uppercase(),
            errors,
            warnings
        )
    };
    Ok(RulesOutput {
        path: input.path.clone(),
        check: check.into(),
        ran,
        errors,
        warnings,
        total,
        violations,
        diagnostics,
        summary,
    })
}
