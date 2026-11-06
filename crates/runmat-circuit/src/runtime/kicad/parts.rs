//! A KiCad import's `part` declarations: each component's library symbol, read into the
//! schematic's pool, becomes a part of the circuit, so the imported `.circuit` compiles on its
//! own, as the other PCB imports' circuits do.

use crate::import::{file_safe, Report};
use crate::ir::{Design, PartRef};
use crate::library::catalog::{CatalogPart, CatalogPin};
use crate::library::{Package, Pool};
use std::collections::BTreeMap;

/// Part-number properties, in the order they're preferred.
const MPN_KEYS: &[&str] = &["MPN", "Manufacturer Part Number", "Manufacturer_Part_Number", "Part Number"];
const MAKER_KEYS: &[&str] = &["Manufacturer", "MFR", "Manufacturer_Name"];

fn property<'a>(props: &'a BTreeMap<String, String>, keys: &[&str]) -> Option<&'a str> {
    keys.iter().find_map(|k| props.iter().find(|(p, v)| p.eq_ignore_ascii_case(k) && !v.trim().is_empty()).map(|(_, v)| v.trim()))
}

/// The board's packages by designator (`R1` → the package the board places), from a board
/// read beside the schematic.
pub(crate) fn board_packages(board: &crate::layout::Board) -> BTreeMap<String, Package> {
    board.footprints.values().map(|f| (f.reference.clone(), f.package.clone())).collect()
}

/// Turns every `PartRef::Entity` of `design` into a `part` declaration (`PartRef::Part`).
///
/// - A part's pins are its symbol's pins, every unit's, named by pin number as the
///   schematic's connections name them, each on the pad of that number.
/// - Its package is the board's footprint for the component (or, for a component the board
///   leaves off, a footprint of the same name the board places), written to
///   `<footprint_dir>/<name>.kicad_mod` (returned as file name → text); otherwise the
///   symbol's footprint name stays as written and the report lists it.
/// - Parts are named by their part-number property, else `<symbol>-<value>` (with the
///   footprint when one symbol and value sit on two footprints).
pub(crate) fn declare_parts(design: &mut Design, pool: &Pool, packages: &BTreeMap<String, Package>, footprint_dir: &str, report: &mut Report) -> BTreeMap<String, String> {
    let mut files = BTreeMap::new();
    let mut parts: BTreeMap<String, CatalogPart> = BTreeMap::new();
    for sheet in design.sheets.values_mut() {
        for c in &mut sheet.components {
            let PartRef::Entity { entity, package } = &c.part else { continue };
            let Some(e) = pool.entity(entity) else {
                report.not_carried(format!("part {}: symbol {entity} has no library symbol in the schematic", c.name));
                continue;
            };
            let mut pins = Vec::new();
            for g in &e.gates {
                let Some(unit) = pool.units.get(&g.unit) else { continue };
                for p in &unit.pins {
                    if !pins.iter().any(|q: &CatalogPin| q.name == p.name) {
                        pins.push(CatalogPin { name: p.name.clone(), pads: vec![p.name.clone()], kind: p.kind.name().into() });
                    }
                }
            }
            if e.gates.len() > 1 {
                report.approximated("symbols of several units are carried as one part with every unit's pins");
            }
            let footprint = package.clone().unwrap_or_default();
            // The board's footprint for the component, else one the board places for another
            // component of the same footprint name (a part the board leaves off).
            let placed = packages.get(&c.name).or_else(|| packages.values().find(|p| !footprint.is_empty() && p.name == footprint));
            let package_ref = match placed {
                Some(pkg) => {
                    let key = file_safe(&pkg.name);
                    files.entry(format!("{footprint_dir}/{key}.kicad_mod")).or_insert_with(|| crate::kicad::footprint::write_mod(pkg));
                    format!("{footprint_dir}/{key}.kicad_mod")
                }
                None => {
                    report.not_carried(format!("footprint {} (footprints come from the board)", if footprint.is_empty() { "(none)" } else { &footprint }));
                    footprint.clone()
                }
            };
            let value = c.properties.get("Value").cloned().unwrap_or_default();
            let fallback = if value.is_empty() { entity.clone() } else { format!("{entity}-{value}") };
            let mut name = property(&c.properties, MPN_KEYS).map(str::to_owned).unwrap_or_else(|| fallback.clone());
            if parts.get(&name).is_some_and(|p| p.package != package_ref) {
                name = format!("{name}-{}", package_ref.rsplit('/').next().unwrap_or(&package_ref).trim_end_matches(".kicad_mod"));
            }
            if name == fallback {
                report.approximated("parts without a part number are named from their symbol and value");
            }
            let datasheet = property(&c.properties, &["Datasheet"]).filter(|d| *d != "~").map(str::to_owned);
            parts.entry(name.clone()).or_insert_with(|| CatalogPart {
                mpn: name.clone(),
                manufacturer: property(&c.properties, MAKER_KEYS).unwrap_or_default().to_owned(),
                description: String::new(),
                category: String::new(),
                prefix: e.prefix.clone(),
                package: package_ref.clone(),
                pins,
                // The component keeps its own properties.
                params: BTreeMap::new(),
                datasheet,
                spice: None,
            });
            c.part = PartRef::Part(name);
        }
    }
    if !parts.is_empty() {
        report.approximated("part symbols are drawn from their pins, not from the file's symbol graphics");
    }
    report.count("parts", parts.len());
    design.parts.extend(parts);
    files
}
