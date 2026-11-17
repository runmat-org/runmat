//! The right-click commands that change what a component is, in the schematic and on the board
//! alike. Each writes the component's statement in the `.circuit`; Update board carries a new
//! part or footprint to the board, and Swap writes both files at once.
//!
//! - `circuit.change-part` (`values.part`): the selected components take another part, by MPN.
//!   Pins map by name, so the new part must have every pin the schematic connects.
//! - `circuit.footprint` (`values.footprint`): another footprint for the part. A component
//!   named by its entity takes the package as written; a component placed as an orderable part
//!   takes the library's part with the same symbol, the same value and that footprint. A
//!   footprint generator string (`R0603`) stands for the package it generates.
//! - `circuit.fitted` (`values.fitting`, `values.part` for an alternate, `values.variant`): the
//!   components' fitting in a variant, written in the variant's `overrides` by component path
//!   when the file declares the variant, else in the component's own `variants`.
//! - `circuit.swap-pins` (`values.swap`, with `values.pins` or `values.gates`): two pins of one
//!   gate, or two gates, of the selected component trade connections. The library allows a
//!   swap only between pins (or gates) with the same non-zero swap group. The schematic's
//!   connections follow and the board's pads trade nets, in every channel the sheet repeats.

use super::edit::{failure, EditError, EditOutput, EditRequest};
use super::set_property::{circuit_change, own_component, text};
use super::*;
use crate::ir::{Endpoint, Fitting, PartRef, SheetDef};
use crate::library::{Entity, Part, ResolvedPin};

pub(super) const COMMANDS: &[&str] = &[
    "circuit.change-part",
    "circuit.footprint",
    "circuit.fitted",
    "circuit.swap-pins",
];

/// Params that make a part's value: a footprint change keeps them.
const VALUE_KEYS: &[&str] = &["resistance", "capacitance", "inductance"];

pub(super) fn run(project: &Project, request: &EditRequest) -> Result<EditOutput, EditError> {
    match request.command.as_str() {
        "circuit.change-part" => change_part(project, request),
        "circuit.footprint" => footprint(project, request),
        "circuit.fitted" => fitted(project, request),
        _ => swap(project, request),
    }
}

fn components(request: &EditRequest) -> Result<Vec<&str>, EditError> {
    let out: Vec<&str> = request
        .selection
        .iter()
        .filter_map(|id| id.strip_prefix("component:"))
        .collect();
    if out.is_empty() || out.len() != request.selection.len() {
        return Err(EditError::new("select the components first"));
    }
    Ok(out)
}

fn value(request: &EditRequest, field: &str, label: &str) -> Result<String, EditError> {
    request
        .values
        .get(field)
        .and_then(text)
        .ok_or_else(|| EditError::new(format!("{label} can't be empty")))
}

fn def_and_index<'d>(
    own: &'d mut crate::ir::Design,
    sheet: &str,
    local: &str,
) -> (&'d mut SheetDef, usize) {
    let def = own.sheets.get_mut(sheet).expect("found by own_component");
    let at = def
        .components
        .iter()
        .position(|c| c.name == local)
        .expect("found by own_component");
    (def, at)
}

/// Every pin name a component's connections in its sheet write (`1`, `A.IN+`).
fn connected_pins(def: &SheetDef, local: &str) -> Vec<String> {
    let mut out: Vec<String> = def
        .connections
        .iter()
        .flat_map(|c| c.ends.iter())
        .chain(def.no_erc.iter())
        .filter_map(|e| match e {
            Endpoint::Pin { component, pin } if component == local => Some(pin.clone()),
            _ => None,
        })
        .collect();
    out.sort();
    out.dedup();
    out
}

fn names(pin: &ResolvedPin, gates: usize) -> [String; 2] {
    let full = format!("{}.{}", pin.gate, pin.pin);
    [
        if gates == 1 {
            pin.pin.clone()
        } else {
            full.clone()
        },
        full,
    ]
}

fn pool_part<'p>(project: &'p Project, mpn: &str) -> Result<(&'p Part, &'p Entity), EditError> {
    let part = project.pool.part(mpn).ok_or_else(|| {
        EditError::new(format!(
            "no part `{mpn}` in the file or the library; search the library for it"
        ))
    })?;
    let entity = project
        .pool
        .entities
        .get(&part.entity)
        .ok_or_else(|| EditError::new(format!("{mpn}'s symbol isn't in the library")))?;
    Ok((part, entity))
}

fn change_part(project: &Project, request: &EditRequest) -> Result<EditOutput, EditError> {
    let mpn = value(request, "part", "Part")?;
    let (part, entity) = pool_part(project, &mpn)?;
    let pins = project.pool.pins(entity, Some(part));
    let gates = entity.gates.len();
    let mut own = project.own.clone();
    let mut done = Vec::new();
    for designator in components(request)? {
        let (sheet, local) = own_component(project, designator)?;
        let (def, at) = def_and_index(&mut own, &sheet, &local);
        let missing: Vec<String> = connected_pins(def, &local)
            .into_iter()
            .filter(|p| {
                !pins
                    .iter()
                    .any(|r| names(r, gates).contains(p) || &r.pin == p)
            })
            .collect();
        if !missing.is_empty() {
            return Err(EditError::new(format!(
                "{mpn} has no pin {}, which {designator} connects; pins map by name",
                missing.join(", ")
            )));
        }
        if def.components[at].part == PartRef::Part(mpn.clone()) {
            continue;
        }
        def.components[at].part = PartRef::Part(mpn.clone());
        done.push(designator);
    }
    if done.is_empty() {
        return Ok(unchanged(format!("the components are {mpn} already")));
    }
    circuit_change(
        project,
        &own,
        format!(
            "{} part {mpn}; Update board to carry it to the board",
            done.join(", ")
        ),
    )
}

fn unchanged(summary: String) -> EditOutput {
    EditOutput {
        summary,
        files: Vec::new(),
        report: None,
    }
}

fn package_name(project: &Project, part: &Part) -> String {
    project
        .pool
        .packages
        .get(&part.package)
        .map(|p| p.name.clone())
        .unwrap_or_default()
}

fn footprint(project: &Project, request: &EditRequest) -> Result<EditOutput, EditError> {
    let written = value(request, "footprint", "Footprint")?;
    // A generator string (`R0603`) names the package it generates (`R_0603_1608Metric_IPCN`).
    let package = crate::library::footprint::from_string(&written).map_or(written, |p| p.name);
    let mut own = project.own.clone();
    let mut done = Vec::new();
    for designator in components(request)? {
        let (sheet, local) = own_component(project, designator)?;
        let (def, at) = def_and_index(&mut own, &sheet, &local);
        match def.components[at].part.clone() {
            PartRef::Entity { package: p, entity } => {
                if p.as_deref() == Some(package.as_str()) {
                    continue;
                }
                def.components[at].part = PartRef::Entity {
                    entity,
                    package: Some(package.clone()),
                };
            }
            PartRef::Part(mpn) => {
                let (current, _) = pool_part(project, &mpn)?;
                if package_name(project, current).eq_ignore_ascii_case(&package) {
                    continue;
                }
                let same_value = |p: &Part| {
                    let (a, b) = (
                        project.pool.effective_part(current),
                        project.pool.effective_part(p),
                    );
                    VALUE_KEYS.iter().all(|k| {
                        a.params.get(*k).map(|v| &v.value) == b.params.get(*k).map(|v| &v.value)
                    })
                };
                let siblings: Vec<&Part> = project
                    .pool
                    .parts
                    .values()
                    .filter(|p| p.entity == current.entity)
                    .collect();
                let pick = siblings.iter().copied().find(|p: &&Part| {
                    package_name(project, p).eq_ignore_ascii_case(&package) && same_value(p)
                });
                let Some(pick) = pick else {
                    let mut offered: Vec<String> = siblings
                        .iter()
                        .filter(|p: &&&Part| same_value(p))
                        .map(|p: &&Part| package_name(project, p))
                        .collect();
                    offered.sort();
                    offered.dedup();
                    return Err(EditError::new(format!(
                        "the library has no {mpn} equivalent in {package}; {designator}'s value comes in {}",
                        offered.join(", ")
                    )));
                };
                def.components[at].part = PartRef::Part(pick.mpn.clone());
            }
        }
        done.push(designator);
    }
    if done.is_empty() {
        return Ok(unchanged(format!("the components use {package} already")));
    }
    circuit_change(
        project,
        &own,
        format!(
            "{} footprint {package}; Update board to carry it to the board",
            done.join(", ")
        ),
    )
}

fn fitted(project: &Project, request: &EditRequest) -> Result<EditOutput, EditError> {
    let variant = request
        .values
        .get("variant")
        .and_then(text)
        .or_else(|| request.options.variant.clone())
        .or_else(|| {
            (project.design.variants.len() == 1)
                .then(|| project.design.variants.keys().next().cloned())
                .flatten()
        })
        .ok_or_else(|| {
            EditError::new(if project.design.variants.is_empty() {
                "the circuit has no variants; add one to fit parts per variant".to_owned()
            } else {
                "choose the variant; fitting is set per variant".to_owned()
            })
        })?;
    if !project.design.variants.contains_key(&variant) {
        return Err(EditError::new(format!(
            "the circuit has no variant `{variant}`"
        )));
    }
    let fitting = match request.values.get("fitting").and_then(text).as_deref() {
        Some("fitted") | None => Fitting::Fitted,
        Some("not_fitted") => Fitting::NotFitted,
        Some("alternate") => {
            let mpn = value(request, "part", "Alternate part")?;
            pool_part(project, &mpn)?;
            Fitting::Alternate {
                part: Some(PartRef::Part(mpn)),
                params: Default::default(),
            }
        }
        Some(other) => {
            return Err(EditError::new(format!(
                "`{other}` isn't a fitting; expected fitted, not_fitted or alternate"
            )))
        }
    };
    let mut own = project.own.clone();
    let mut done = Vec::new();
    for designator in components(request)? {
        let path = project
            .netlist
            .components
            .iter()
            .find(|c| c.designator == designator)
            .map(|c| c.path.to_string())
            .ok_or_else(|| EditError::new(format!("{designator} isn't in the design")))?;
        if let Some(def) = own.variants.get_mut(&variant) {
            // The variant's overrides name the component by path; a designator key goes.
            def.overrides.remove(designator);
            let before = def.overrides.get(&path).cloned();
            if fitting == Fitting::Fitted && def.base.is_none() {
                def.overrides.remove(&path);
            } else {
                def.overrides.insert(path.clone(), fitting.clone());
            }
            if def.overrides.get(&path) != before.as_ref() {
                done.push(designator);
            }
        } else {
            let (sheet, local) = own_component(project, designator)?;
            let (def, at) = def_and_index(&mut own, &sheet, &local);
            let variants = &mut def.components[at].variants;
            let before = variants.get(&variant).cloned();
            if fitting == Fitting::Fitted {
                variants.remove(&variant);
            } else {
                variants.insert(variant.clone(), fitting.clone());
            }
            if variants.get(&variant) != before.as_ref() {
                done.push(designator);
            }
        }
    }
    if done.is_empty() {
        return Ok(unchanged(format!(
            "the components have that fitting in {variant} already"
        )));
    }
    let word = match &fitting {
        Fitting::Fitted => "fitted".to_owned(),
        Fitting::NotFitted => "not fitted".to_owned(),
        Fitting::Alternate { part, .. } => match part {
            Some(PartRef::Part(mpn)) => format!("fitted as {mpn}"),
            _ => "an alternate".to_owned(),
        },
    };
    circuit_change(
        project,
        &own,
        format!("{} {word} in {variant}", done.join(", ")),
    )
}

/// Two names from `values.<field>` (`1, 2`, `A B`).
fn pair(request: &EditRequest, field: &str, label: &str) -> Result<(String, String), EditError> {
    let written = value(request, field, label)?;
    let parts: Vec<&str> = written
        .split([',', ' ', '/'])
        .filter(|s| !s.is_empty())
        .collect();
    match parts.as_slice() {
        [a, b] if a != b => Ok((a.to_string(), b.to_string())),
        _ => Err(EditError::new(format!(
            "{label} takes two names, such as {}",
            if field == "gates" { "A, B" } else { "1, 2" }
        ))),
    }
}

fn swap(project: &Project, request: &EditRequest) -> Result<EditOutput, EditError> {
    let picked = components(request)?;
    let [designator] = picked[..] else {
        return Err(EditError::new(
            "select one component to swap its pins or gates",
        ));
    };
    let comp = project
        .netlist
        .components
        .iter()
        .find(|c| c.designator == designator)
        .ok_or_else(|| EditError::new(format!("{designator} isn't in the design")))?;
    let (entity, part) = match (
        comp.entity.and_then(|e| project.pool.entities.get(&e)),
        comp.part.and_then(|p| project.pool.parts.get(&p)),
    ) {
        (Some(e), p) => (e, p),
        _ => {
            return Err(EditError::new(format!(
                "{designator} has no library symbol to swap on"
            )))
        }
    };
    let gates = entity.gates.len();
    let pins = project.pool.pins(entity, part);
    let unit_pin = |gate: &str, pin: &str| {
        entity
            .gates
            .iter()
            .find(|g| g.name == gate)
            .and_then(|g| project.pool.units.get(&g.unit))
            .and_then(|u| u.pins.iter().find(|p| p.name == pin))
    };
    // Pairs of resolved pins that trade connections.
    let mut trades: Vec<(&ResolvedPin, &ResolvedPin)> = Vec::new();
    let gate_swap = request.values.get("swap").and_then(Value::as_str) == Some("gates");
    if gate_swap {
        let (a, b) = pair(request, "gates", "Gates")?;
        let find = |name: &str| {
            entity
                .gates
                .iter()
                .find(|g| g.name == name)
                .ok_or_else(|| EditError::new(format!("{designator} has no gate {name}")))
        };
        let (ga, gb) = (find(&a)?, find(&b)?);
        if ga.swap_group == 0 || ga.swap_group != gb.swap_group || ga.unit != gb.unit {
            return Err(EditError::new(format!(
                "gates {a} and {b} of {designator} aren't interchangeable in its library part"
            )));
        }
        for pa in pins.iter().filter(|p| p.gate == a) {
            let pb = pins
                .iter()
                .find(|p| p.gate == b && p.pin == pa.pin)
                .expect("same unit");
            trades.push((pa, pb));
        }
    } else {
        let (a, b) = pair(request, "pins", "Pins")?;
        let find = |name: &str| {
            pins.iter()
                .find(|p| names(p, gates).iter().any(|n| n == name))
                .ok_or_else(|| EditError::new(format!("{designator} has no pin {name}")))
        };
        let (pa, pb) = (find(&a)?, find(&b)?);
        let group = |p: &ResolvedPin| unit_pin(&p.gate, &p.pin).map_or(0, |u| u.swap_group);
        if pa.gate != pb.gate || group(pa) == 0 || group(pa) != group(pb) {
            return Err(EditError::new(format!(
                "pins {a} and {b} of {designator} aren't interchangeable in its library part"
            )));
        }
        trades.push((pa, pb));
    }
    // The schematic: the sheet's connections trade the pins' names.
    let (sheet, local) = own_component(project, designator)?;
    let mut own = project.own.clone();
    let def = own.sheets.get_mut(&sheet).expect("found by own_component");
    let renamed = |pin: &str| -> Option<String> {
        trades.iter().find_map(|(a, b)| {
            let (na, nb) = (names(a, gates), names(b, gates));
            if na.iter().any(|n| n == pin) {
                Some(nb[if pin.contains('.') { 1 } else { 0 }].clone())
            } else if nb.iter().any(|n| n == pin) {
                Some(na[if pin.contains('.') { 1 } else { 0 }].clone())
            } else {
                None
            }
        })
    };
    let mut moved = 0;
    for end in def
        .connections
        .iter_mut()
        .flat_map(|c| c.ends.iter_mut())
        .chain(def.no_erc.iter_mut())
    {
        if let Endpoint::Pin { component, pin } = end {
            if *component == local {
                if let Some(to) = renamed(pin) {
                    *pin = to;
                    moved += 1;
                }
            }
        }
    }
    if moved == 0 {
        return Err(EditError::new(format!(
            "{designator}'s swapped pins aren't connected in the schematic"
        )));
    }
    let what = if gate_swap { "gates" } else { "pins" };
    let label = trades
        .first()
        .map(|(a, b)| {
            if gate_swap {
                format!("{} and {}", a.gate, b.gate)
            } else {
                format!("{} and {}", names(a, gates)[0], names(b, gates)[0])
            }
        })
        .unwrap_or_default();
    let mut out = circuit_change(
        project,
        &own,
        format!("{designator} {what} {label} swapped"),
    )?;
    // The board: every channel's footprint of the component trades its pads' nets.
    if let Some(layout) = &project.layout {
        let drawn: Vec<String> = project
            .netlist
            .components
            .iter()
            .filter(|c| c.path.leaf() == local.as_str() && c.sheet == comp.sheet)
            .map(|c| c.designator.clone())
            .collect();
        let (change, n) =
            board_edit::edit_parsed(&layout.path, &layout.text, &layout.parsed, |b| {
                let mut n = 0;
                for f in b
                    .footprints
                    .values_mut()
                    .filter(|f| drawn.contains(&f.reference))
                {
                    for (from, to) in &trades {
                        for (pa, pb) in from.pads.iter().zip(&to.pads) {
                            let (na, nb) = (f.pad_nets.remove(pa), f.pad_nets.remove(pb));
                            if let Some(net) = nb {
                                f.pad_nets.insert(pa.clone(), net);
                            }
                            if let Some(net) = na {
                                f.pad_nets.insert(pb.clone(), net);
                            }
                            n += 1;
                        }
                    }
                }
                Ok(n)
            })
            .map_err(failure)?;
        if n > 0 {
            out.files
                .push(file_change(&layout.path, &layout.text, change));
        }
    }
    Ok(out)
}
