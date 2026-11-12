//! The checks a library `.circuit` passes on its own: each footprint's pads, courtyard and
//! drills, and each part's pins against its footprint and symbol. Every finding names the
//! footprint, part or symbol it is about, and the pad or pin.
//!
//! - pad numbers are unique within a footprint;
//! - the courtyard is closed (the reader refuses an open one) and contains every pad;
//! - a drill is smaller than its pad;
//! - every part pin lands on a pad its footprint has;
//! - every pad has a pin, or the part lists it in `nc`;
//! - a part's pins and its symbol's pins match by name.

use super::catalog::PartRecord;
use super::{Package, PadDef, PadKind, Symbol};
use crate::geom::poly::PolySet;
use crate::geom::{to_mm, Point};
use std::collections::BTreeSet;

/// What is wrong with footprint `name`, one finding per problem.
pub fn footprint_findings(name: &str, package: &Package) -> Vec<String> {
    let mut out = Vec::new();
    let mut seen = BTreeSet::new();
    for pad in &package.pads {
        // A mechanical hole may go unnumbered; numbered pads are unique.
        if pad.name.is_empty() && pad.kind == PadKind::Npth {
            continue;
        }
        if !seen.insert(pad.name.as_str()) {
            out.push(format!(
                "footprint `{name}`: pad `{}` is declared twice",
                pad.name
            ));
        }
    }
    if package.courtyard.is_empty() {
        out.push(format!("footprint `{name}` has no courtyard"));
    } else {
        let courtyard = PolySet::from_rings(package.courtyard.clone());
        for pad in &package.pads {
            if let Some(corner) = pad_corners(pad)
                .into_iter()
                .find(|c| courtyard.contains(*c) == Some(false))
            {
                out.push(format!(
                    "footprint `{name}`: pad `{}` reaches past the courtyard at [{}, {}]",
                    pad.name,
                    to_mm(corner.x),
                    to_mm(corner.y)
                ));
            }
        }
    }
    for pad in &package.pads {
        let Some(drill) = &pad.drill else { continue };
        if pad.kind == PadKind::Npth {
            continue;
        }
        let (w, h) = pad.shape.size();
        let hole = drill.diameter.max(drill.slot.unwrap_or(0));
        let across = if drill.slot.is_some() {
            w.max(h)
        } else {
            w.min(h)
        };
        if drill.diameter >= w.min(h) || hole >= across {
            out.push(format!(
                "footprint `{name}`: pad `{}` has a {} mm drill in a {} x {} mm pad; a drill is smaller than its pad",
                pad.name,
                to_mm(hole),
                to_mm(w),
                to_mm(h)
            ));
        }
    }
    out
}

/// The corners of a pad's bounding box, turned by its rotation (a pad on a courtyard edge is
/// inside).
fn pad_corners(pad: &PadDef) -> Vec<Point> {
    let (w, h) = pad.shape.size();
    [(-1i64, -1i64), (1, -1), (1, 1), (-1, 1)]
        .iter()
        .map(|(sx, sy)| Point::new(sx * w / 2, sy * h / 2).rotate(pad.rotation) + pad.at)
        .collect()
}

/// What is wrong with part `part` on its footprint `package`: pins on pads it lacks, and pads
/// with no pin that the part doesn't list in `nc`.
pub fn part_findings(part: &PartRecord, package: &Package) -> Vec<String> {
    let mut out = Vec::new();
    let mut used = BTreeSet::new();
    for pin in &part.pins {
        for pad in &pin.pads {
            used.insert(pad.as_str());
            if package.pad(pad).is_none() {
                out.push(format!(
                    "part `{}`: pin `{}` lands on pad `{pad}`, which its footprint has no pad of",
                    part.mpn, pin.name
                ));
            }
        }
    }
    for pad in &part.nc {
        if package.pad(pad).is_none() {
            out.push(format!(
                "part `{}`: `nc` lists pad `{pad}`, which its footprint has no pad of",
                part.mpn
            ));
        }
        if used.contains(pad.as_str()) {
            out.push(format!(
                "part `{}`: pad `{pad}` has a pin and is also listed in `nc`",
                part.mpn
            ));
        }
    }
    let nc: BTreeSet<&str> = part.nc.iter().map(String::as_str).collect();
    let open: Vec<&str> = package
        .pads
        .iter()
        .filter(|p| !p.name.is_empty() && p.kind != PadKind::Npth)
        .map(|p| p.name.as_str())
        .filter(|p| !used.contains(p) && !nc.contains(p))
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    if !open.is_empty() {
        out.push(format!(
            "part `{}`: its footprint has {} pads and its pins land on {}; pad{} {} {} no pin (map {} to a pin or list {} in `nc`)",
            part.mpn,
            package.pads.iter().filter(|p| p.kind != PadKind::Npth).count(),
            used.len(),
            if open.len() == 1 { "" } else { "s" },
            open.iter().map(|p| format!("`{p}`")).collect::<Vec<_>>().join(", "),
            if open.len() == 1 { "has" } else { "have" },
            if open.len() == 1 { "it" } else { "each" },
            if open.len() == 1 { "it" } else { "them" },
        ));
    }
    out
}

/// Whether part `part`'s pins and symbol `name`'s pins match by name.
pub fn pins_match_symbol(part: &PartRecord, name: &str, symbol: &Symbol) -> Result<(), String> {
    let pins: BTreeSet<&str> = part.pins.iter().map(|p| p.name.as_str()).collect();
    let drawn: BTreeSet<&str> = symbol.pins.iter().map(|p| p.pin.as_str()).collect();
    if let Some(p) = pins.difference(&drawn).next() {
        return Err(format!(
            "part `{}`: pin `{p}` has no pin of that name on symbol `{name}`",
            part.mpn
        ));
    }
    if let Some(p) = drawn.difference(&pins).next() {
        return Err(format!(
            "part `{}`: symbol `{name}` draws pin `{p}`, which the part has no pin of",
            part.mpn
        ));
    }
    Ok(())
}

#[cfg(test)]
#[path = "checks_tests.rs"]
mod tests;
