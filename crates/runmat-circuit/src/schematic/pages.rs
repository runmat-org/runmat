//! A sheet instance's second page: the unused units of its large parts.
//!
//! An FPGA's unused IO banks are as tall as the used ones, so on the instance's own page they
//! push the frame up a size or two and shrink everything at fit. Hand-drawn schematics keep
//! them on a page of their own, and so does the layout:
//!
//! - A unit is **unused** when it is one of a split gate's units ([`super::units`]) and every
//!   pin is unconnected: no net, a net with only that pin on it, or a pin marked `no_erc`.
//! - When a page has unused units and other content, the unused units go to page 2 of the same
//!   instance (`page = 2`, same `path`), titled "unused units". A unit that becomes used moves
//!   back to page 1, and its recorded place, which is in page-2 coordinates, is dropped.
//! - The KiCad export writes one `.kicad_sch` per instance, so [`merge_pages`] puts page 2 to the
//!   right of page 1 on one sheet of paper.

use super::{shift, SchematicIndex, Unit};
use crate::elab::Netlist;

/// What page 2 holds, as its tab and title block say it.
pub const SPARE_TITLE: &str = "unused units";

/// Whether a split gate's unit has no connection at all (see the module docs).
pub(super) fn unused(nl: &Netlist, comp: usize, unit: &Unit) -> bool {
    let c = &nl.components[comp];
    unit.label.is_some()
        && !unit.pins.is_empty()
        && unit.pins.iter().all(|&p| {
            nl.no_erc.contains(&(comp, p))
                || c.pins[p].net.is_none_or(|n| nl.nets[n].pins.len() <= 1)
        })
}

/// One page per instance, as a KiCad hierarchy has it: each instance's page 2 sits to the right
/// of its page 1 on one sheet of user-sized paper, tops aligned.
pub fn merge_pages(index: &SchematicIndex) -> SchematicIndex {
    let mut out = SchematicIndex::default();
    for sheet in &index.sheets {
        if sheet.page == 1 {
            out.sheets.push(sheet.clone());
            continue;
        }
        let Some(first) = out.sheets.iter_mut().rev().find(|s| s.path == sheet.path) else {
            out.sheets.push(sheet.clone());
            continue;
        };
        let height = first.height_mm.max(sheet.height_mm);
        let lift = height - first.height_mm;
        shift(first, 0.0, lift);
        let mut extra = sheet.clone();
        shift(&mut extra, first.width_mm, height - sheet.height_mm);
        first.width_mm += sheet.width_mm;
        first.height_mm = height;
        first.frame = "User".to_owned();
        first.symbols.extend(extra.symbols);
        first.sheet_symbols.extend(extra.sheet_symbols);
        first.wires.extend(extra.wires);
        first.junctions.extend(extra.junctions);
        first.labels.extend(extra.labels);
        first.no_connects.extend(extra.no_connects);
    }
    out
}

pub(super) fn one() -> u32 {
    1
}

pub(super) fn is_one(v: &u32) -> bool {
    *v == 1
}
