//! Pipe-record documents, from the format's public description: a schematic or board is a
//! list of records, each a line of `|KEY=value|KEY=value|` pairs, with `RECORD` naming its
//! kind. The ASCII files hold one record per line; the binary files hold the same records in
//! a compound-file container ([`binary`]).
//!
//! - Schematic records are numbered (`RECORD=1` a part, `2` a pin, `27` a wire …). A record
//!   that belongs to a part names it with `OWNERINDEX`, the part record's index among the
//!   records after the header. Schematic lengths are in 10 mil units, with an optional
//!   `_FRAC` field in hundred-thousandths of a unit; y grows up.
//! - Board records are named (`RECORD=Component`, `Pad`, `Track` …). Items name their net and
//!   component by index among the `Net` and `Component` records (`NET=3`, `COMPONENT=0`).
//!   Lengths carry their unit (`1000mil`, `2.54mm`); a bare number is in mil; y grows up.
//! - The import writes `<stem>.circuit` (a `part` declaration per part, pins on the pads
//!   their numbers name), the board's footprints as `<stem>-footprints/<pattern>.kicad_mod`,
//!   and with a board `<stem>.circuit.layout` using the circuit.

pub mod binary;
pub mod board;
pub mod primitives;
pub mod schematic;

use super::{Imported, Report};
use crate::geom::{mm, Coord};
use crate::runtime::{CIRCUIT_SUFFIX, LAYOUT_SUFFIX};
use std::collections::BTreeMap;

/// The format's names in reports: its file extensions, ASCII or binary.
pub const FORMAT_ASCII: &str = ".SchDoc / .PcbDoc (ASCII)";
pub const FORMAT_BINARY: &str = ".SchDoc / .PcbDoc (binary)";

/// One record: its fields by upper-case key.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Record(pub BTreeMap<String, String>);

impl Record {
    /// Parses `|KEY=value|KEY=value|`. Empty fields and fields without `=` are skipped.
    pub fn parse(line: &str) -> Record {
        let mut out = BTreeMap::new();
        for field in line.trim().trim_matches('\0').split('|') {
            if let Some((k, v)) = field.split_once('=') {
                let k = k.trim();
                if !k.is_empty() {
                    out.insert(k.to_ascii_uppercase(), v.to_owned());
                }
            }
        }
        Record(out)
    }
    pub fn kind(&self) -> &str {
        self.str("RECORD")
    }
    pub fn str(&self, key: &str) -> &str {
        self.0.get(key).map(String::as_str).unwrap_or("")
    }
    pub fn has(&self, key: &str) -> bool {
        self.0.contains_key(key)
    }
    /// A plain number (`3`, `-1.5`); 0 when absent.
    pub fn num(&self, key: &str) -> f64 {
        self.str(key).trim().parse().unwrap_or(0.0)
    }
    /// An index field; `None` when absent or negative.
    pub fn index(&self, key: &str) -> Option<usize> {
        let v: i64 = self.str(key).trim().parse().ok()?;
        (v >= 0).then_some(v as usize)
    }
    pub fn flag(&self, key: &str) -> bool {
        matches!(self.str(key).trim().to_ascii_uppercase().as_str(), "T" | "TRUE" | "1")
    }
    /// A schematic length in 10 mil units, with its `<key>_FRAC` part.
    pub fn sch(&self, key: &str) -> f64 {
        self.num(key) + self.num(&format!("{key}_FRAC")) / 100_000.0
    }
    /// A board length with its unit (`10mil`, `0.25mm`, a bare number in mil), in board units.
    pub fn len(&self, key: &str) -> Coord {
        board_length(self.str(key))
    }
}

/// A board length with its unit, in board units.
pub fn board_length(s: &str) -> Coord {
    let s = s.trim().to_ascii_lowercase();
    let (n, scale) = if let Some(v) = s.strip_suffix("mil") {
        (v, 0.0254)
    } else if let Some(v) = s.strip_suffix("mm") {
        (v, 1.0)
    } else if let Some(v) = s.strip_suffix("in") {
        (v, 25.4)
    } else {
        (s.as_str(), 0.0254)
    };
    mm(n.trim().parse::<f64>().unwrap_or(0.0) * scale)
}

/// One schematic unit (10 mil) in millimetres.
pub const SCH_UNIT_MM: f64 = 0.254;

/// A document's records, the header first when it has one.
pub fn records(text: &str) -> Vec<Record> {
    text.lines().filter(|l| l.trim_start().starts_with('|')).map(Record::parse).filter(|r| !r.0.is_empty()).collect()
}

/// Which document a set of records is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Schematic,
    Board,
}

/// A schematic's records are numbered, a board's named.
pub fn kind_of(records: &[Record]) -> Option<Kind> {
    let r = records.iter().find(|r| r.has("RECORD"))?;
    Some(if r.kind().trim().chars().all(|c| c.is_ascii_digit()) { Kind::Schematic } else { Kind::Board })
}

/// Imports the schematic and board documents of `files` (file name → their records) to
/// `<target_stem>.circuit` and `.circuit.layout`. `format` names the format in the report.
pub fn import_records(files: &BTreeMap<String, Vec<Record>>, target_stem: &str, format: &str) -> Result<Imported, String> {
    let mut sheets = Vec::new();
    let mut brd = None;
    for (name, recs) in files {
        match kind_of(recs) {
            Some(Kind::Schematic) => sheets.push((name.clone(), recs)),
            Some(Kind::Board) if brd.is_none() => brd = Some(recs),
            Some(Kind::Board) => return Err(format!("{name}: give one board")),
            None => return Err(format!("{name} has no `|RECORD=…|` lines, so it isn't an ASCII .SchDoc or .PcbDoc document")),
        }
    }
    if sheets.is_empty() && brd.is_none() {
        return Err("give a schematic or board document".into());
    }
    let name = target_stem.rsplit('/').next().unwrap_or(target_stem).to_owned();
    let folder = target_stem.rsplit_once('/').map(|(d, _)| format!("{d}/")).unwrap_or_default();
    let footprint_dir = format!("{name}-footprints");
    let mut report = Report::new(format);
    let mut out = BTreeMap::new();
    let board = match brd {
        Some(recs) => Some(board::read(recs, &name, &mut report)?),
        None => None,
    };
    if !sheets.is_empty() {
        let (packages, footprints) = board.as_ref().map(|b| (b.packages.clone(), b.footprints.clone())).unwrap_or_default();
        let design = schematic::read(&sheets, &name, &footprint_dir, &packages, &footprints, &mut report)?;
        out.insert(format!("{target_stem}{CIRCUIT_SUFFIX}"), crate::syntax::print_circuit(&design).map_err(|e| format!("{e:?}"))?);
    } else {
        report.not_carried("the schematic: only the board was given, so the circuit has no parts");
    }
    if let Some(b) = board {
        let uses = [runmat_design_syntax::document::UseItem {
            symbol: Some(("circuit".into(), name.clone())),
            from: String::new(),
            alias: None,
            revision: None,
            span: Default::default(),
        }];
        out.insert(format!("{target_stem}{LAYOUT_SUFFIX}"), crate::syntax::board::print_board(&b.board, &uses));
        for (key, pkg) in &b.packages {
            out.insert(format!("{folder}{footprint_dir}/{key}.kicad_mod"), crate::kicad::footprint::write_mod(pkg));
        }
    }
    Ok(Imported { files: out, report })
}

/// Imports ASCII pipe-record documents (file name → text).
pub fn import(files: &BTreeMap<String, String>, target_stem: &str) -> Result<Imported, String> {
    let given = files.iter().map(|(n, t)| (n.clone(), records(t))).collect();
    import_records(&given, target_stem, FORMAT_ASCII)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn records_read_their_fields_and_lengths() {
        let r = Record::parse("|RECORD=Pad|X=1000mil|Y=2.54mm|HOLESIZE=40|NAME=1|PLATED=TRUE|");
        assert_eq!(r.kind(), "Pad");
        assert_eq!(r.len("X"), mm(25.4));
        assert_eq!(r.len("Y"), mm(2.54));
        assert_eq!(r.len("HOLESIZE"), mm(1.016));
        assert!(r.flag("PLATED"));
        let s = Record::parse("|RECORD=2|LOCATION.X=100|LOCATION.X_FRAC=50000|OWNERINDEX=3|");
        assert_eq!(s.sch("LOCATION.X"), 100.5);
        assert_eq!(s.index("OWNERINDEX"), Some(3));
        assert_eq!(kind_of(&[s]), Some(Kind::Schematic));
        assert_eq!(kind_of(&[r]), Some(Kind::Board));
    }
}
