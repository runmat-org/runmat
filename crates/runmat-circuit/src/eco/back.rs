//! Back-annotation (what the schematic needs to match the board) and agent change sets.

use super::*;
use crate::elab::{elaborate, ElabOptions};
use crate::ids::{DerivedUuids, LockSection};
use crate::ir::{Design, PartRef};
use crate::layout::Board;
use crate::library::Pool;
use crate::review::ModelEdit;
use std::collections::BTreeSet;

/* ------------------------------------------------------------------------------------------------
 * Back-annotation
 * --------------------------------------------------------------------------------------------- */

/// Board → schematic: designator renumbering, pin and gate swaps and footprint changes made on
/// the board, as changes to the schematic.
pub fn back_annotate(s: &Schematic, board: &Board, o: &ComparatorOptions) -> EcoPlan {
    let nl = s.netlist;
    let root = s.design.root.clone();
    let links = link(s, board, o);
    let mut b = Builder {
        changes: Vec::new(),
        direction: Direction::Back,
    };
    for (ci, fid) in &links.by_comp {
        let c = &nl.components[*ci];
        let f = &board.footprints[fid];
        let key = c.path.to_string();
        let synced = board.sync.components.get(&key).or_else(|| {
            f.component
                .as_ref()
                .and_then(|p| board.sync.components.get(p))
        });
        let (sheet_name, _) = sheet_of(&c.path, &root);
        let sheet_def = c.sheet.clone();
        if f.reference != c.designator && synced.is_none_or(|b| b.designator != f.reference) {
            let both =
                synced.is_some_and(|b| b.designator != c.designator && b.designator != f.reference);
            let ch = b.push(
                ChangeKind::ChangeDesignator,
                &key,
                f.reference.clone(),
                Some(c.designator.clone()),
                Some(f.reference.clone()),
                comp_cause(c, &root, format!("renumbered {} on the board", f.reference)),
                Impact {
                    footprints: vec![f.reference.clone()],
                    ..Impact::default()
                },
                Op::SchDesignator {
                    path: key.clone(),
                    sheet: sheet_def.clone(),
                    to: f.reference.clone(),
                },
            );
            if both {
                ch.conflict = Some(format!(
                    "renamed on both sides since the last sync ({} in the schematic, {} on the board)",
                    c.designator, f.reference
                ));
            }
        }
        let pkg = package_name(s.pool, c.package);
        if !o.ignore_footprints
            && pkg.as_deref().is_some_and(|p| p != f.package.name)
            && synced.is_none_or(|b| b.package != f.package.name)
        {
            let both = synced
                .is_some_and(|b| Some(&b.package) != pkg.as_ref() && b.package != f.package.name);
            // A part of the same entity on the board's package, or the package itself for an
            // entity reference.
            let decl_part = s
                .design
                .sheets
                .get(&sheet_def)
                .and_then(|sd| sd.components.iter().find(|d| d.name == c.path.leaf()))
                .map(|d| d.part.clone());
            let alt = c.entity.and_then(|e| {
                let pkg_uuid = s
                    .pool
                    .packages
                    .values()
                    .find(|p| p.name == f.package.name)?
                    .uuid;
                s.pool
                    .parts
                    .values()
                    .filter(|p| p.entity == e && p.package == pkg_uuid)
                    .map(|p| p.mpn.clone())
                    .min()
            });
            let ch = b.push(
                ChangeKind::ChangeFootprint,
                &key,
                f.reference.clone(),
                pkg.clone(),
                Some(f.package.name.clone()),
                comp_cause(
                    c,
                    &root,
                    format!("{} uses {} on the board", f.reference, f.package.name),
                ),
                Impact {
                    footprints: vec![f.reference.clone()],
                    ..Impact::default()
                },
                Op::SchFootprint {
                    path: key.clone(),
                    sheet: sheet_def.clone(),
                    component: c.path.leaf().to_owned(),
                    part: alt.clone(),
                    package: f.package.name.clone(),
                },
            );
            if both {
                ch.conflict = Some(format!(
                    "footprint changed on both sides since the last sync ({} and {})",
                    pkg.clone().unwrap_or_default(),
                    f.package.name
                ));
            }
            match (&decl_part, &alt) {
                (Some(PartRef::Entity { .. }), _) => {}
                (_, Some(mpn)) => ch
                    .issues
                    .push((Status::Warning, format!("the part changes to {mpn}"))),
                _ => ch.issues.push((
                    Status::Error,
                    format!(
                        "no part in the library puts this component on {}",
                        f.package.name
                    ),
                )),
            }
        }
        // Swaps: pin pairs whose board nets are each other's schematic nets.
        let want = sch_pads(nl, c);
        let board_net = |pad: &str| f.pad_nets.get(pad).map(|n| board.net_name(*n).to_owned());
        let pin_of = |pad: &str| c.pins.iter().find(|p| p.pads.iter().any(|x| x == pad));
        let net_of_pin = |p: &crate::elab::PinInst, board_side: bool| -> Option<String> {
            let pad = p.pads.first()?;
            if board_side {
                board_net(pad)
            } else {
                want.get(pad).cloned().flatten()
            }
        };
        // Whole-gate swaps first: every pin of gate A carries gate B's schematic net and back.
        let gates: BTreeSet<&str> = c
            .pins
            .iter()
            .map(|p| p.gate.as_str())
            .filter(|g| !g.is_empty())
            .collect();
        let gates: Vec<&str> = gates.into_iter().collect();
        let mut gate_swapped: BTreeSet<String> = BTreeSet::new();
        for (i, ga) in gates.iter().enumerate() {
            for gb in gates.iter().skip(i + 1) {
                if gate_swapped.contains(*ga) || gate_swapped.contains(*gb) {
                    continue;
                }
                let pins_a: Vec<&crate::elab::PinInst> =
                    c.pins.iter().filter(|p| p.gate == *ga).collect();
                let partner = |pa: &crate::elab::PinInst| {
                    c.pins.iter().find(|p| p.gate == *gb && p.pin == pa.pin)
                };
                let same_shape = pins_a.len() == c.pins.iter().filter(|p| p.gate == *gb).count()
                    && pins_a.iter().all(|pa| partner(pa).is_some());
                if !same_shape {
                    continue;
                }
                let differs = pins_a
                    .iter()
                    .any(|pa| net_of_pin(pa, true) != net_of_pin(pa, false));
                let swapped = pins_a.iter().all(|pa| {
                    let pb = partner(pa).unwrap();
                    net_of_pin(pa, true) == net_of_pin(pb, false)
                        && net_of_pin(pb, true) == net_of_pin(pa, false)
                });
                if !(differs && swapped) {
                    continue;
                }
                gate_swapped.insert((*ga).to_owned());
                gate_swapped.insert((*gb).to_owned());
                let mut impact = Impact::default();
                for pa in &pins_a {
                    let pb = partner(pa).unwrap();
                    for pad in pa.pads.iter().chain(pb.pads.iter()) {
                        impact.merge(copper_at_pad(board, *fid, pad));
                    }
                }
                let ch = b.push(
                    ChangeKind::SwapGates,
                    &format!("{key}#{ga}#{gb}"),
                    format!("{} gates {ga} \u{2194} {gb}", f.reference),
                    Some(format!("gate {ga}")),
                    Some(format!("gate {gb}")),
                    comp_cause(
                        c,
                        &root,
                        format!(
                            "gates {ga} and {gb} of {} were swapped on the board",
                            f.reference
                        ),
                    ),
                    impact,
                    Op::SchSwapGates {
                        path: key.clone(),
                        sheet: sheet_def.clone(),
                        component: c.path.leaf().to_owned(),
                        a: (*ga).to_owned(),
                        b: (*gb).to_owned(),
                    },
                );
                if c.channel.is_some() {
                    ch.issues.push((Status::Warning, format!("sheet {} repeats; the swap changes every channel, not only {sheet_name}", c.sheet)));
                }
            }
        }
        // Single pin swaps among the rest, preferring partners of the same pin name.
        let pads: Vec<&String> = want
            .keys()
            .filter(|p| pin_of(p).is_none_or(|x| !gate_swapped.contains(&x.gate)))
            .collect();
        let mut used: BTreeSet<&String> = BTreeSet::new();
        for a in &pads {
            if used.contains(*a) {
                continue;
            }
            let (wa, ba) = (want[*a].clone(), board_net(a));
            if wa == ba || wa.is_none() {
                continue;
            }
            let fits = |bp: &&&String| {
                !used.contains(**bp)
                    && *bp != a
                    && want[**bp] != board_net(bp)
                    && wa == board_net(bp)
                    && want[**bp] == ba
            };
            let same_name = pads
                .iter()
                .filter(fits)
                .find(|bp| pin_of(bp).map(|p| &p.pin) == pin_of(a).map(|p| &p.pin));
            let Some(bp) = same_name.or_else(|| pads.iter().find(fits)) else {
                continue;
            };
            used.insert(a);
            used.insert(bp);
            let (Some(pa), Some(pb)) = (pin_of(a), pin_of(bp)) else {
                continue;
            };
            let (qa, qb) = (qualified(pa), qualified(pb));
            let mut impact = copper_at_pad(board, *fid, a);
            impact.merge(copper_at_pad(board, *fid, bp));
            let ch = b.push(
                ChangeKind::SwapPins,
                &format!("{key}#{a}#{bp}"),
                format!("{}.{a} \u{2194} {}.{bp}", f.reference, f.reference),
                Some(format!(
                    "{qa} on {}",
                    net_label(&wa.clone().unwrap_or_default())
                )),
                Some(format!(
                    "{qa} on {}",
                    net_label(&want[*bp].clone().unwrap_or_default())
                )),
                comp_cause(
                    c,
                    &root,
                    format!(
                        "pins {a} and {bp} of {} were swapped on the board",
                        f.reference
                    ),
                ),
                impact,
                Op::SchSwapPins {
                    path: key.clone(),
                    sheet: sheet_def.clone(),
                    component: c.path.leaf().to_owned(),
                    a: qa.clone(),
                    b: qb.clone(),
                },
            );
            if pa.kind != pb.kind || pa.gate != pb.gate {
                ch.issues.push((
                    Status::Warning,
                    format!("{qa} and {qb} are not equivalent pins"),
                ));
            }
            if c.channel.is_some() {
                ch.issues.push((
                    Status::Warning,
                    format!("sheet {} repeats; the swap changes every channel", c.sheet),
                ));
            }
        }
    }
    finish(Direction::Back, b.changes, o)
}

pub(super) fn qualified(p: &crate::elab::PinInst) -> String {
    if p.gate.is_empty() {
        p.pin.clone()
    } else {
        format!("{}.{}", p.gate, p.pin)
    }
}

/* ------------------------------------------------------------------------------------------------
 * Agent change sets
 * --------------------------------------------------------------------------------------------- */

/// An agent's model edits as an ECO: applied to a copy of the design, elaborated with the lock,
/// compared with the board. Every change carries the agent as its origin.
#[allow(clippy::too_many_arguments)]
pub fn agent_change_set(
    design: &Design,
    edits: &[ModelEdit],
    title: &str,
    pool: &Pool,
    lock: &LockSection,
    board: &Board,
    elab: &ElabOptions,
    o: &ComparatorOptions,
) -> Result<(Design, EcoPlan), String> {
    let mut next = design.clone();
    for e in edits {
        crate::review::apply(&mut next, e)?;
    }
    let mut lock = lock.clone();
    let nl = elaborate(&next, pool, elab, Some(&lock));
    lock.reconcile(&nl.lock_items(), &mut DerivedUuids { namespace: 0x5eed });
    let s = Schematic {
        design: &next,
        netlist: &nl,
        pool,
        lock: &lock,
    };
    let mut plan = compare(&s, board, o);
    for c in &mut plan.changes {
        c.origin = Origin::Agent {
            title: title.to_owned(),
        };
    }
    Ok((next, plan))
}
