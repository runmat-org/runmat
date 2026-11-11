//! Applying back-annotation lines to the schematic and the lock, and recording the sync.

use super::*;
use crate::ids::{LockSection, NamePath};
use crate::ir::{Design, Endpoint, PartRef};
use crate::layout::Board;
use std::collections::BTreeSet;

/* ------------------------------------------------------------------------------------------------
 * Back-annotation apply
 * --------------------------------------------------------------------------------------------- */

/// Executes the selected back-annotation changes on the schematic (and the lock, which holds
/// auto-annotated designators), all or nothing. The board's sync record takes the new state so
/// the next forward compare is clean for these items.
pub fn apply_back(
    design: &mut Design,
    lock: &mut LockSection,
    board: &mut Board,
    plan: &EcoPlan,
    selected: &BTreeSet<String>,
) -> Result<Vec<String>, ApplyError> {
    let checks = validate(plan, board, selected);
    let failures: Vec<(String, String)> = checks
        .iter()
        .filter(|(_, c)| c.status == Status::Error)
        .map(|(id, c)| (id.clone(), c.reason().unwrap_or_default()))
        .collect();
    if !failures.is_empty() {
        return Err(ApplyError { failures });
    }
    let mut next = design.clone();
    let mut next_lock = lock.clone();
    let mut applied = Vec::new();
    let mut errors = Vec::new();
    for c in plan.changes.iter().filter(|c| selected.contains(&c.id)) {
        let r = match &c.op {
            Op::SchDesignator { path, sheet, to } => {
                back_designator(&mut next, &mut next_lock, path, sheet, to)
            }
            Op::SchSwapPins {
                sheet,
                component,
                a,
                b,
                ..
            } => swap_endpoints(&mut next, sheet, component, |p| {
                if p == a {
                    Some(b.clone())
                } else if p == b {
                    Some(a.clone())
                } else {
                    None
                }
            }),
            Op::SchSwapGates {
                sheet,
                component,
                a,
                b,
                ..
            } => {
                let (pa, pb) = (format!("{a}."), format!("{b}."));
                swap_endpoints(&mut next, sheet, component, |p| {
                    p.strip_prefix(&pa)
                        .map(|r| format!("{pb}{r}"))
                        .or_else(|| p.strip_prefix(&pb).map(|r| format!("{pa}{r}")))
                })
            }
            Op::SchFootprint {
                sheet,
                component,
                part,
                package,
                ..
            } => {
                match next
                    .sheets
                    .get_mut(sheet)
                    .and_then(|s| s.components.iter_mut().find(|d| &d.name == component))
                {
                    Some(d) => match (&mut d.part, part) {
                        (PartRef::Entity { package: p, .. }, _) => {
                            *p = Some(package.clone());
                            Ok(())
                        }
                        (r, Some(mpn)) => {
                            *r = PartRef::Part(mpn.clone());
                            Ok(())
                        }
                        _ => Err(format!("no part puts {component} on {package}")),
                    },
                    None => Err(format!("{component} is not declared on sheet {sheet}")),
                }
            }
            _ => Ok(()),
        };
        match r {
            Ok(()) => applied.push(c.id.clone()),
            Err(e) => errors.push((c.id.clone(), e)),
        }
    }
    if !errors.is_empty() {
        return Err(ApplyError { failures: errors });
    }
    // The board already holds these states: record them as synced.
    for c in plan.changes.iter().filter(|c| selected.contains(&c.id)) {
        let path = match &c.op {
            Op::SchDesignator { path, .. }
            | Op::SchSwapPins { path, .. }
            | Op::SchSwapGates { path, .. }
            | Op::SchFootprint { path, .. } => path,
            _ => continue,
        };
        let Some(f) = board
            .footprints
            .values()
            .find(|f| f.component.as_deref() == Some(path))
        else {
            continue;
        };
        let pads = f
            .pad_nets
            .iter()
            .map(|(p, n)| (p.clone(), board.net_name(*n).to_owned()))
            .collect();
        let rec = SyncedComponent {
            designator: f.reference.clone(),
            package: f.package.name.clone(),
            pads,
        };
        board.sync.components.insert(path.clone(), rec);
    }
    *design = next;
    *lock = next_lock;
    Ok(applied)
}

pub(super) fn back_designator(
    design: &mut Design,
    lock: &mut LockSection,
    path: &str,
    sheet: &str,
    to: &str,
) -> Result<(), String> {
    let p = NamePath::parse(path);
    let repeated = p.0.iter().any(|s| s.ends_with(']'));
    let decl_name = p.leaf().to_owned();
    let auto = repeated
        || design
            .sheets
            .get(sheet)
            .and_then(|s| s.components.iter().find(|d| d.name == decl_name))
            .is_none_or(|d| d.name.contains('?'));
    if auto {
        // Auto-annotated (`R?`) and repeated-channel designators live in the lock.
        let e = lock
            .entries
            .get_mut(&p)
            .ok_or_else(|| format!("{path} is not in the lock"))?;
        e.designator = Some(to.to_owned());
        return Ok(());
    }
    // An explicitly named component: rename it, keeping its identity (`renamed_from`).
    let s = design
        .sheets
        .get_mut(sheet)
        .ok_or_else(|| format!("no sheet {sheet}"))?;
    let d = s
        .components
        .iter_mut()
        .find(|d| d.name == decl_name)
        .ok_or_else(|| format!("{path} is not declared"))?;
    let old = std::mem::replace(&mut d.name, to.to_owned());
    d.renamed_from = Some(old.clone());
    for conn in s.connections.iter_mut() {
        for e in conn.ends.iter_mut() {
            if let Endpoint::Pin { component, .. } = e {
                if *component == old {
                    *component = to.to_owned();
                }
            }
        }
    }
    Ok(())
}

pub(super) fn swap_endpoints(
    design: &mut Design,
    sheet: &str,
    component: &str,
    map: impl Fn(&str) -> Option<String>,
) -> Result<(), String> {
    let s = design
        .sheets
        .get_mut(sheet)
        .ok_or_else(|| format!("no sheet {sheet}"))?;
    let mut hits = 0;
    for conn in s.connections.iter_mut() {
        for e in conn.ends.iter_mut() {
            if let Endpoint::Pin { component: c, pin } = e {
                if c == component {
                    if let Some(n) = map(pin) {
                        *pin = n;
                        hits += 1;
                    }
                }
            }
        }
    }
    if hits == 0 {
        return Err(format!("no connections to swap on {component}"));
    }
    Ok(())
}

/// Records the whole schematic as synced (the state after a full ECO), for boards made before
/// the sync record existed.
pub fn record_sync(board: &mut Board, s: &Schematic) {
    let o = ComparatorOptions::default();
    let links = link(s, board, &o);
    for (ci, fid) in &links.by_comp {
        let c = &s.netlist.components[*ci];
        let f = &board.footprints[fid];
        board.sync.components.insert(
            c.path.to_string(),
            SyncedComponent {
                designator: f.reference.clone(),
                package: f.package.name.clone(),
                pads: f
                    .pad_nets
                    .iter()
                    .map(|(p, n)| (p.clone(), board.net_name(*n).to_owned()))
                    .collect(),
            },
        );
    }
}
