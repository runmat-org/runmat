//! The overview's state lines, by kind, each from one check.

use super::{plural, Fact, Level, Ran};
use crate::check::{check, CheckInput, CheckOutput};
use crate::inspect::{inspect, InspectInput};
use crate::rules::{drc, erc, RulesInput, RulesOutput};
use crate::{DesignKind, Files, ToolError};

fn fact(id: &str, label: &str, value: String, level: Level, source: &str) -> Fact {
    Fact {
        id: id.into(),
        label: label.into(),
        value,
        level,
        source: source.into(),
    }
}

/// `No errors`, `2 errors, 1 warning`.
pub(super) fn problems(errors: usize, warnings: usize) -> (String, Level) {
    match (errors, warnings) {
        (0, 0) => ("No errors or warnings".into(), Level::Ok),
        (0, w) => (
            format!("No errors, {}", plural(w, "warning", "warnings")),
            Level::Warning,
        ),
        (e, 0) => (plural(e, "error", "errors"), Level::Error),
        (e, w) => (
            format!(
                "{}, {}",
                plural(e, "error", "errors"),
                plural(w, "warning", "warnings")
            ),
            Level::Error,
        ),
    }
}

fn rules(path: &str) -> RulesInput {
    RulesInput {
        path: path.to_owned(),
        severities: None,
        filter: None,
        limit: Some(0),
    }
}

/// The first error's message, for a line that names what failed.
fn first_error(c: &CheckOutput) -> Option<&str> {
    c.diagnostics
        .iter()
        .find(|d| d.severity == "error" && d.cause == "root")
        .map(|d| d.message.as_str())
}

fn count(c: &CheckOutput, key: &str) -> usize {
    c.counts.get(key).copied().unwrap_or(0)
}

/// What the file is and its state lines.
pub(super) fn state(
    kind: DesignKind,
    path: &str,
    files: &Files,
    ran: &mut Ran,
) -> Result<(String, Vec<Fact>), ToolError> {
    let checked = check(
        &CheckInput {
            path: path.to_owned(),
        },
        files,
    )?;
    ran.note("design.check");
    let mut out = Vec::new();
    let about = match kind {
        DesignKind::Part => part(&checked, &mut out),
        DesignKind::Assembly => assembly(&checked, &mut out),
        DesignKind::Drawing => drawing(&checked, &mut out),
        DesignKind::Circuit | DesignKind::Layout => {
            circuit(kind, path, files, &checked, &mut out, ran)
        }
    };
    Ok((about, out))
}

fn regeneration(c: &CheckOutput, label: &str, out: &mut Vec<Fact>) {
    let (mut value, level) = problems(c.errors, c.warnings);
    if let Some(first) = first_error(c) {
        value = format!("{value}: {first}");
    }
    out.push(fact("regeneration", label, value, level, "design.check"));
}

fn part(c: &CheckOutput, out: &mut Vec<Fact>) -> String {
    let features = count(c, "features");
    let failed = c
        .features
        .iter()
        .filter(|f| f.status == "failed" || f.status == "blocked")
        .count();
    regeneration(c, "Regeneration", out);
    if features > 0 {
        let value = if failed == 0 {
            format!("{} regenerate", plural(features, "feature", "features"))
        } else {
            format!("{failed} of {features} features fail or are blocked")
        };
        out.push(fact(
            "features",
            "Features",
            value,
            if failed == 0 { Level::Ok } else { Level::Error },
            "design.check",
        ));
    }
    let inputs: Vec<String> = c
        .inputs
        .iter()
        .map(|i| match &i.display {
            Some(d) => format!("{} {d}", i.name),
            None => format!("{} unresolved", i.name),
        })
        .collect();
    if !inputs.is_empty() {
        let unresolved = c.inputs.iter().any(|i| i.display.is_none());
        out.push(fact(
            "inputs",
            "Inputs",
            inputs.join(", "),
            if unresolved {
                Level::Error
            } else {
                Level::Info
            },
            "design.check",
        ));
    }
    let mut what = vec![
        plural(features, "feature", "features"),
        plural(count(c, "bodies"), "body", "bodies"),
    ];
    if count(c, "regions") > 0 {
        what.push(plural(count(c, "regions"), "named region", "named regions"));
    }
    format!("Part: {}", what.join(", "))
}

fn assembly(c: &CheckOutput, out: &mut Vec<Fact>) -> String {
    regeneration(c, "Assembly check", out);
    format!(
        "Assembly: {} of {}, {}",
        plural(count(c, "occurrences"), "occurrence", "occurrences"),
        plural(count(c, "prototypes"), "part", "parts"),
        plural(count(c, "mates"), "mate", "mates")
    )
}

fn drawing(c: &CheckOutput, out: &mut Vec<Fact>) -> String {
    regeneration(c, "Drawing check", out);
    let annotations = count(c, "annotations");
    out.push(fact(
        "annotations",
        "Annotations",
        if annotations == 0 {
            "No dimensions or notes yet".into()
        } else {
            plural(annotations, "dimension or note", "dimensions and notes")
        },
        if annotations == 0 {
            Level::Warning
        } else {
            Level::Info
        },
        "design.check",
    ));
    format!(
        "Drawing: {} on {}",
        plural(count(c, "views"), "view", "views"),
        plural(count(c, "sheets"), "sheet", "sheets")
    )
}

fn circuit(
    kind: DesignKind,
    path: &str,
    files: &Files,
    c: &CheckOutput,
    out: &mut Vec<Fact>,
    ran: &mut Ran,
) -> String {
    if c.errors > 0 && c.counts.is_empty() {
        regeneration(c, "Compile", out);
        return format!("{}: doesn't compile", noun(kind));
    }
    if let Ok(e) = erc(&rules(path), files) {
        ran.note("design.erc");
        push_rules(out, "erc", "Electrical rules", &e, "design.erc");
    }
    // Whether the circuit has a board comes from the inspection, which loads the layout.
    let has_board = board(path, files, out, ran);
    if has_board {
        if let Ok(d) = drc(&rules(path), files) {
            ran.note("design.drc");
            if d.ran {
                push_rules(out, "drc", "Design rules", &d, "design.drc");
            }
        }
    } else if kind == DesignKind::Circuit {
        out.push(fact(
            "board_sync",
            "Board",
            "No board yet".into(),
            Level::Info,
            "design.check",
        ))
    }
    simulations(path, files, out, ran);
    let mut about = format!(
        "{}: {}, {} on {}",
        noun(kind),
        plural(count(c, "components"), "component", "components"),
        plural(count(c, "nets"), "net", "nets"),
        plural(count(c, "sheets"), "sheet", "sheets")
    );
    if has_board {
        about.push_str("; with a board");
    }
    about
}

fn noun(kind: DesignKind) -> &'static str {
    if kind == DesignKind::Layout {
        "Board"
    } else {
        "Schematic"
    }
}

fn push_rules(out: &mut Vec<Fact>, id: &str, label: &str, r: &RulesOutput, source: &str) {
    if !r.ran {
        out.push(fact(
            id,
            label,
            "Didn't run: the files don't compile".into(),
            Level::Error,
            source,
        ));
        return;
    }
    let (value, level) = problems(r.errors, r.warnings);
    out.push(fact(id, label, value, level, source));
}

/// Board sync and placement, from the inspection the guides read. False when there is no board.
fn board(path: &str, files: &Files, out: &mut Vec<Fact>, ran: &mut Ran) -> bool {
    let input = InspectInput {
        path: path.to_owned(),
        sections: Some(vec!["eco".into(), "unplaced".into()]),
        filter: None,
        density: None,
        set: None,
        limit: Some(0),
        changed: None,
        rays: None,
    };
    let Ok(seen) = inspect(&input, files) else {
        return false;
    };
    ran.note("design.inspect");
    let Some(eco) = seen.totals.get("eco differences").copied() else {
        return false;
    };
    out.push(fact(
        "board_sync",
        "Board sync",
        if eco == 0 {
            "The board has every schematic change".into()
        } else {
            format!(
                "{} from the schematic not on the board yet",
                plural(eco, "change", "changes")
            )
        },
        if eco == 0 { Level::Ok } else { Level::Warning },
        "design.inspect",
    ));
    let unplaced = seen.totals.get("unplaced components").copied().unwrap_or(0);
    out.push(fact(
        "placement",
        "Placement",
        if unplaced == 0 {
            "Every fitted part is on the board".into()
        } else {
            format!("{} not on the board", plural(unplaced, "part", "parts"))
        },
        if unplaced == 0 {
            Level::Ok
        } else {
            Level::Warning
        },
        "design.inspect",
    ));
    true
}

/// The design's `simulation` sections and what each probes, from the circuit compiler.
fn simulations(path: &str, files: &Files, out: &mut Vec<Fact>, ran: &mut Ran) {
    let Some(text) = files.get(path) else { return };
    let compiled = runmat_circuit::runtime::compile(
        path,
        text,
        files,
        &crate::circuit_values::circuit_options(path, text, files),
    );
    let Some(index) = compiled.index else { return };
    ran.note("circuit.simulations");
    let value = if index.simulations.is_empty() {
        "No simulation section".to_owned()
    } else {
        index
            .simulations
            .iter()
            .map(|s| format!("{} ({})", s.name, plural(s.probes.len(), "probe", "probes")))
            .collect::<Vec<_>>()
            .join(", ")
    };
    out.push(fact(
        "simulation",
        "Simulation",
        value,
        Level::Info,
        "circuit.simulations",
    ));
}
