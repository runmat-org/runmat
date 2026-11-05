//! ERC and DRC as structured violations: rule, objects, layer, location and source range.

use super::*;

/// Where a finding's source is: a file, its UTF-16 range, and the one-based line and column of
/// the range's start.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceRange {
    pub path: String,
    pub start: usize,
    pub end: usize,
    pub line: usize,
    pub column: usize,
}

impl SourceRange {
    /// The range of `key` in `text` (the text of `key.file`).
    pub fn of(key: &runmat_rules::SourceKey, text: &str) -> Self {
        let start = (key.start as usize).min(text.len());
        let head = &text[..floor_char(text, start)];
        let line = head.matches('\n').count() + 1;
        let column = head.rsplit('\n').next().map_or(0, |l| l.chars().count()) + 1;
        SourceRange {
            path: key.file.clone(),
            start: utf16_offset(text, start),
            end: utf16_offset(text, key.end as usize),
            line,
            column,
        }
    }
}

/// One electrical (ERC) or design-rule (DRC) violation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Violation {
    /// `erc` or `drc`.
    pub check: String,
    pub code: String,
    /// `error`, `warning` or `info`.
    pub severity: String,
    pub message: String,
    /// The DRC rule that set the limit (`rule power-width`), or the ERC check.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rule: Option<String>,
    /// The objects involved: designators, `U3.4` pins and pads, nets, tracks, vias, pours.
    pub items: Vec<String>,
    /// The copper layer (DRC), by its stack-up name.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub layer: Option<String>,
    /// Where on the board (DRC), in millimetres.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub at_mm: Option<[f64; 2]>,
    /// The measured value and the rule's required value, in [`Violation::unit`].
    #[serde(skip_serializing_if = "Option::is_none")]
    pub actual: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub required: Option<f64>,
    /// `mm` for lengths, `count` for counted findings (islands of an unrouted net, spokes).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub unit: Option<String>,
    /// The declaration or rule the finding points back to.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<SourceRange>,
}

/// ERC over the schematic and, when the circuit has a board, zone fill and DRC over it.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct RuleCheckOutput {
    /// The files compiled, so ERC (and DRC, with a board) ran.
    pub compiled: bool,
    pub erc: Vec<Violation>,
    /// `None` when the circuit has no board yet.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub drc: Option<Vec<Violation>>,
    /// Errors that stopped the check (the files don't compile).
    pub diagnostics: Vec<Diagnostic>,
}

fn severity_text(s: &crate::elab::Severity) -> &'static str {
    match s {
        crate::elab::Severity::Error => "error",
        crate::elab::Severity::Warning => "warning",
        crate::elab::Severity::Info => "info",
    }
}

/// Runs ERC and, with a board, DRC on the circuit `path` belongs to: every finding with its
/// rule, the objects involved, where it is and the source it points back to.
pub fn rule_check(
    path: &str,
    text: &str,
    files: &Files,
    options: &CompileOptions,
) -> RuleCheckOutput {
    let project = match load(path, text, files, options) {
        Ok(p) => p,
        Err(diagnostics) => {
            return RuleCheckOutput {
                compiled: false,
                erc: Vec::new(),
                drc: None,
                diagnostics,
            }
        }
    };
    let text_of = |file: &str| -> Option<&str> {
        if file == path {
            Some(text)
        } else if file == project.circuit_path {
            Some(project.circuit_text.as_str())
        } else if let Some(l) = project.layout.as_ref().filter(|l| l.path == file) {
            Some(l.text.as_str())
        } else {
            files.get(file).map(String::as_str)
        }
    };
    let source_of = |key: &runmat_rules::SourceKey| -> Option<SourceRange> {
        if key.file.is_empty() {
            return None;
        }
        text_of(&key.file).map(|t| SourceRange::of(key, t))
    };
    let mut erc_findings = project.netlist.diagnostics.clone();
    erc_findings.extend(erc::run(&project.netlist, &ErcOptions::default()));
    let erc = erc_findings
        .iter()
        .map(|d| Violation {
            check: "erc".into(),
            code: d.code.clone(),
            severity: severity_text(&d.severity).into(),
            message: d.message.clone(),
            rule: None,
            unit: None,
            items: std::iter::once(d.item.clone())
                .chain(d.related.iter().cloned())
                .filter(|s| !s.is_empty())
                .collect(),
            layer: None,
            at_mm: None,
            actual: None,
            required: None,
            source: source_of(&d.source),
        })
        .collect();
    let mut diagnostics = Vec::new();
    let drc = project.layout.as_ref().and_then(|layout| {
        let mut board = layout.parsed.board.clone()?;
        let (mut rules, errors) = BoardRules::new(&board.limits, &board.rules);
        for e in errors {
            diagnostics.push(Diagnostic::error(
                &layout.path,
                Some(&layout.text),
                0,
                0,
                format!("{e:?}"),
            ));
        }
        rules.rebuild(&BoardGraph::new(&board, &[]));
        let fills = fill_zones(&mut board, &rules, &[]);
        let run = Drc::run(&board, &rules, &[], &fills);
        let names = board.stackup.copper_names();
        let nm = |v: f64| v / 1e6;
        // Findings whose values count things (islands, spokes) rather than measure lengths.
        let counted = |code: &str| matches!(code, "unrouted" | "min-spokes");
        Some(
            run.findings()
                .into_iter()
                .map(|f| Violation {
                    check: "drc".into(),
                    code: f.code.clone(),
                    severity: severity_text(&f.severity).into(),
                    message: f.message.clone(),
                    rule: (!f.rule.is_empty()).then(|| f.rule.clone()),
                    items: f
                        .items
                        .iter()
                        .map(|it| crate::index::item_label(&board, it))
                        .collect(),
                    layer: f.layer.and_then(|l| names.get(l as usize).cloned()),
                    at_mm: Some([nm(f.at.x as f64), nm(f.at.y as f64)]),
                    actual: f.actual.map(|v| if counted(&f.code) { v } else { nm(v) }),
                    required: f.required.map(|v| if counted(&f.code) { v } else { nm(v) }),
                    unit: f
                        .actual
                        .or(f.required)
                        .map(|_| if counted(&f.code) { "count" } else { "mm" }.to_owned()),
                    source: source_of(&f.source),
                })
                .collect(),
        )
    });
    RuleCheckOutput {
        compiled: true,
        erc,
        drc,
        diagnostics,
    }
}
