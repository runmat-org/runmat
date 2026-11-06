//! EAGLE XML projects (version 6 and later): a schematic (`.sch`), a board (`.brd`) or both.
//!
//! The import writes `<stem>.circuit` (with a `part` declaration per part number), each
//! package as `<name>-footprints/<library>_<package>.kicad_mod` beside it, and with a board
//! `<stem>.circuit.layout` using the circuit. The binary files of earlier versions aren't read.

pub mod board;
pub mod library;
pub mod schematic;

use super::{xml, Imported, Report};
use crate::runtime::{CIRCUIT_SUFFIX, LAYOUT_SUFFIX};
use std::collections::BTreeMap;

fn drawing_of(name: &str, text: &str) -> Result<xml::Element, String> {
    if !text.trim_start().starts_with('<') {
        return Err(format!("{name} isn't EAGLE XML (files from before version 6 are binary and aren't read)"));
    }
    let root = xml::parse(text).map_err(|e| format!("{name}: {e}"))?;
    if root.name != "eagle" {
        return Err(format!("{name} isn't an EAGLE file (its root is <{}>)", root.name));
    }
    root.child("drawing").cloned().ok_or_else(|| format!("{name} has no <drawing>"))
}

/// Imports the `.sch` and `.brd` of `files` (file name → text) to `<target_stem>.circuit` and
/// `.circuit.layout`.
pub fn import(files: &BTreeMap<String, String>, target_stem: &str) -> Result<Imported, String> {
    let lower = |k: &String, ext: &str| k.to_ascii_lowercase().ends_with(ext);
    let sch = files.iter().find(|(k, _)| lower(k, ".sch"));
    let brd = files.iter().find(|(k, _)| lower(k, ".brd"));
    if sch.is_none() && brd.is_none() {
        return Err("give an EAGLE schematic (.sch) or board (.brd)".into());
    }
    let name = target_stem.rsplit('/').next().unwrap_or(target_stem).to_owned();
    let folder = target_stem.rsplit_once('/').map(|(d, _)| format!("{d}/")).unwrap_or_default();
    let footprint_dir = format!("{name}-footprints");
    let mut report = Report::new("EAGLE XML");
    let mut out = BTreeMap::new();
    let mut packages = BTreeMap::new();
    if let Some((file, text)) = sch {
        let drawing = drawing_of(file, text)?;
        let read = schematic::read(&drawing, &name, &footprint_dir, &mut report)?;
        let circuit = crate::syntax::print_circuit(&read.design).map_err(|e| format!("{e:?}"))?;
        out.insert(format!("{target_stem}{CIRCUIT_SUFFIX}"), circuit);
        packages.extend(read.packages);
    }
    if let Some((file, text)) = brd {
        let drawing = drawing_of(file, text)?;
        let board = board::read(&drawing, &name, &mut report)?;
        for f in board.footprints.values() {
            packages.entry(f.package.name.clone()).or_insert_with(|| f.package.clone());
        }
        let uses = [runmat_design_syntax::document::UseItem {
            symbol: Some(("circuit".into(), name.clone())),
            from: String::new(),
            alias: None,
            revision: None,
            span: Default::default(),
        }];
        out.insert(format!("{target_stem}{LAYOUT_SUFFIX}"), crate::syntax::board::print_board(&board, &uses));
    }
    for (key, pkg) in &packages {
        out.insert(format!("{folder}{footprint_dir}/{key}.kicad_mod"), crate::kicad::footprint::write_mod(pkg));
    }
    Ok(Imported { files: out, report })
}
