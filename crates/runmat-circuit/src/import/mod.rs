//! Imports from other PCB tools into RunMat sources, written clean-room from the formats'
//! public descriptions: a `.circuit` (with a `part` declaration for each part it uses), a
//! `.circuit.layout` when the project has a board, and the footprints the parts name as
//! `.kicad_mod` files beside them. Each import returns a [`Report`] of what it carried, what
//! it approximated and what it left out.
//!
//! - [`xml_project`]: XML project schematics (`.sch`) and boards (`.brd`), version 6 and later.
//! - [`shape_json`]: shape-string JSON schematics and boards (`.json`, standard edition).
//! - [`pipe_record`]: pipe-record schematics and boards (`.SchDoc`, `.PcbDoc`), ASCII and binary.

pub mod pipe_record;
pub mod shape_json;
pub mod xml;
pub mod xml_project;

use serde::Serialize;
use std::collections::BTreeMap;

/// What an import carried over, what it approximated and what it left out.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Report {
    /// The format read, named by its file extensions as users know them (`.sch / .brd`).
    pub format: String,
    /// Items carried over, by kind (`components`, `nets`, `tracks` …).
    pub carried: BTreeMap<String, usize>,
    /// Items carried with a change, each with how often it occurred.
    pub approximated: BTreeMap<String, usize>,
    /// Items left out, each with how often it occurred.
    pub not_carried: BTreeMap<String, usize>,
}

impl Report {
    pub fn new(format: &str) -> Self {
        Report { format: format.to_owned(), ..Default::default() }
    }
    pub fn count(&mut self, kind: &str, n: usize) {
        *self.carried.entry(kind.to_owned()).or_default() += n;
    }
    pub fn approximated(&mut self, what: impl Into<String>) {
        *self.approximated.entry(what.into()).or_default() += 1;
    }
    pub fn not_carried(&mut self, what: impl Into<String>) {
        *self.not_carried.entry(what.into()).or_default() += 1;
    }
    /// The report as lines of text: what was carried, then each approximation and omission.
    pub fn lines(&self) -> Vec<String> {
        let mut out = vec![format!(
            "Imported from {}: {}",
            self.format,
            self.carried.iter().map(|(k, n)| format!("{n} {k}")).collect::<Vec<_>>().join(", ")
        )];
        let times = |n: &usize| if *n > 1 { format!(" (×{n})") } else { String::new() };
        for (what, n) in &self.approximated {
            out.push(format!("Approximated: {what}{}", times(n)));
        }
        for (what, n) in &self.not_carried {
            out.push(format!("Not carried over: {what}{}", times(n)));
        }
        out
    }
}

/// RunMat sources written by an import, with its report.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Imported {
    /// Workspace path → text.
    pub files: BTreeMap<String, String>,
    pub report: Report,
}

/// The circuit written for a board imported alone: an empty circuit, so the layout's
/// `use circuit` resolves and the board compiles as written. Its parts stay board-only until a
/// schematic is given, which the report says.
pub fn circuit_stub(name: &str, report: &mut Report) -> Result<String, String> {
    let mut stub = crate::ir::Design::new(name);
    stub.sheet_mut("main");
    report.not_carried("the schematic: only the board was given, so the circuit has no parts");
    crate::syntax::print_circuit(&stub).map_err(|e| format!("{e:?}"))
}

/// A file name made from a library item's name (`passives:0603` → `passives_0603`).
pub fn file_safe(name: &str) -> String {
    let s: String = name.chars().map(|c| if c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.') { c } else { '_' }).collect();
    s.trim_matches('.').to_owned()
}
