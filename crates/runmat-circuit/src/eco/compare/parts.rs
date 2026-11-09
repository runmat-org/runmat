//! Components: placing the ones the board lacks (in their channel's room), removing the ones
//! the schematic dropped, and their values, footprints, pins and nets.

use super::*;
use crate::geom::{Coord, Rect};
use crate::index::{component_value, param_display};
use std::collections::{BTreeMap, BTreeSet};

/// Returns the rooms this phase proposes for channels the board has no room for yet.
pub(super) fn compare_parts(x: &Inputs, nets: &Nets, b: &mut Builder) -> BTreeMap<String, Rect> {
    let Inputs {
        s,
        nl,
        board,
        o,
        root,
        links,
        base,
    } = *x;
    let board_as_sch = |name: &str| nets.board_as_sch(name);
    let add_net_req = |name: &str| nets.add_req(name);
    // ---- Components.
    let mut placer = Placer::new(board);
    let mut adds: BTreeMap<Option<String>, Vec<usize>> = BTreeMap::new();
    for (ci, c) in nl.components.iter().enumerate() {
        if !links.by_comp.contains_key(&ci) {
            adds.entry(if o.rooms_per_channel {
                channel_of(c)
            } else {
                None
            })
            .or_default()
            .push(ci);
        }
    }
    let room_rect = |name: &str| {
        board
            .rooms
            .iter()
            .find(|r| r.name == name)
            .map(|r| r.outline.bbox())
    };
    let mut new_rooms: BTreeMap<String, Rect> = BTreeMap::new();
    for (room, cis) in &adds {
        let sizes: Vec<(Coord, Coord)> = cis
            .iter()
            .map(|ci| package_extent(s.pool, nl.components[*ci].package))
            .collect();
        let (spots, placed_room, staged) = match room.as_deref().and_then(room_rect) {
            Some(area) => {
                let occupied: Vec<Rect> = board
                    .footprints
                    .values()
                    .filter(|f| area.intersects(&f.bbox()))
                    .map(|f| f.bbox())
                    .collect();
                (Placer::grid(area, &sizes, &occupied), room.clone(), false)
            }
            None => {
                let (spots, area) = placer.stage(&sizes);
                if let Some(r) = room {
                    new_rooms.insert(r.clone(), area);
                }
                (spots, room.clone(), true)
            }
        };
        for (k, ci) in cis.iter().enumerate() {
            let c = &nl.components[*ci];
            let at = spots.get(k).copied().unwrap_or(placer.staging);
            let pads: BTreeMap<String, String> = sch_pads(nl, c)
                .into_iter()
                .filter_map(|(p, n)| n.map(|n| (p, n)))
                .collect();
            let pkg = package_name(s.pool, c.package);
            let mut requires: Vec<String> = pads.values().filter_map(|n| add_net_req(n)).collect();
            requires.sort();
            requires.dedup();
            let ch = b.push(
                ChangeKind::AddComponent,
                &c.path.to_string(),
                c.designator.clone(),
                None,
                Some(match &pkg {
                    Some(p) => p.clone(),
                    None => "no footprint".to_owned(),
                }),
                comp_cause(
                    c,
                    root,
                    format!("{} is in the schematic, not on the board", c.designator),
                ),
                Impact {
                    footprints: vec![c.designator.clone()],
                    nets: pads.values().cloned().collect(),
                    placement: Some(Placement {
                        at,
                        room: placed_room.clone(),
                        staged,
                    }),
                    ..Impact::default()
                },
                Op::AddComponent {
                    path: c.path.to_string(),
                    designator: c.designator.clone(),
                    package: c.package,
                    value: component_value(c).unwrap_or_default(),
                    params: c
                        .params
                        .iter()
                        .map(|(k, v)| (k.clone(), param_display(v)))
                        .collect(),
                    pads: pads.clone(),
                    fitted: c.fitted(),
                    at,
                },
            );
            ch.requires = requires;
            match c.package.map(|u| s.pool.packages.get(&u)) {
                None => ch.issues.push((
                    Status::Error,
                    format!("{} has no footprint assigned", c.designator),
                )),
                Some(None) => ch.issues.push((
                    Status::Error,
                    format!("the footprint for {} is not in the library", c.designator),
                )),
                Some(Some(p)) => {
                    let missing: Vec<&str> = pads
                        .keys()
                        .filter(|pad| p.pad(pad).is_none())
                        .map(|s| s.as_str())
                        .collect();
                    if !missing.is_empty() {
                        ch.issues.push((
                            Status::Error,
                            format!("{} has no pad {}", p.name, missing.join(", ")),
                        ));
                    }
                }
            }
        }
    }

    for fid in &links.unlinked {
        let f = &board.footprints[fid];
        if !o.remove_unlinked_footprints
            || f.properties.get("board_only").is_some_and(|v| v == "true")
        {
            continue;
        }
        let impact = footprint_impact(board, *fid);
        let ch = b.push(
            ChangeKind::RemoveComponent,
            &f.reference,
            f.reference.clone(),
            Some(f.package.name.clone()),
            None,
            root_cause(
                root,
                None,
                format!("no schematic component links to {}", f.reference),
            ),
            impact,
            Op::RemoveComponent {
                reference: f.reference.clone(),
            },
        );
        if ch.impact.copper() > 0 {
            ch.issues.push((
                Status::Warning,
                format!("{} routed items lose their pad", ch.impact.copper()),
            ));
        }
        if f.locked {
            ch.issues
                .push((Status::Warning, format!("{} is locked", f.reference)));
        }
    }

    for (ci, fid) in &links.by_comp {
        let c = &nl.components[*ci];
        let f = &board.footprints[fid];
        let key = c.path.to_string();
        let synced = base
            .components
            .get(&key)
            .or_else(|| f.component.as_ref().and_then(|p| base.components.get(p)));
        // Designator.
        if f.reference != c.designator {
            let board_edit = synced.is_some_and(|b| b.designator == c.designator);
            let both =
                synced.is_some_and(|b| b.designator != c.designator && b.designator != f.reference);
            let ch = b.push(
                ChangeKind::ChangeDesignator,
                &key,
                c.designator.clone(),
                Some(f.reference.clone()),
                Some(c.designator.clone()),
                comp_cause(
                    c,
                    root,
                    if c.renamed_from.is_some() {
                        format!(
                            "renamed in the schematic; the board still calls it {}",
                            f.reference
                        )
                    } else {
                        format!("annotated {} in the schematic", c.designator)
                    },
                ),
                Impact {
                    footprints: vec![f.reference.clone()],
                    ..Impact::default()
                },
                Op::SetDesignator {
                    reference: f.reference.clone(),
                    to: c.designator.clone(),
                    path: key.clone(),
                },
            );
            if board_edit {
                ch.issues.push((
                    Status::Warning,
                    format!(
                        "{} was renumbered on the board; back-annotate to keep it",
                        f.reference
                    ),
                ));
            }
            if both {
                ch.conflict = Some(format!(
                    "renamed on both sides since the last sync ({} in the schematic, {} on the board)",
                    c.designator, f.reference
                ));
            }
        }
        // Footprint.
        let pkg = package_name(s.pool, c.package);
        // A package the library doesn't have still differs (and fails validation).
        let differs = match (&c.package, &pkg) {
            (Some(_), Some(p)) => *p != f.package.name,
            (Some(_), None) => true,
            (None, _) => false,
        };
        if !o.ignore_footprints && differs {
            let to = pkg
                .clone()
                .unwrap_or_else(|| "a footprint missing from the library".to_owned());
            let board_edit = synced.is_some_and(|b| b.package == to);
            let both = synced.is_some_and(|b| b.package != to && b.package != f.package.name);
            let mut impact = footprint_impact(board, *fid);
            impact.footprints = vec![f.reference.clone()];
            let ch = b.push(
                ChangeKind::ChangeFootprint,
                &key,
                c.designator.clone(),
                Some(f.package.name.clone()),
                Some(to.clone()),
                comp_cause(c, root, format!("the part of {} uses {to}", c.designator)),
                impact,
                Op::SetFootprint {
                    reference: f.reference.clone(),
                    package: c.package,
                },
            );
            match c.package.and_then(|u| s.pool.packages.get(&u)) {
                None => ch.issues.push((
                    Status::Error,
                    format!("the footprint for {} is not in the library", c.designator),
                )),
                Some(p) => {
                    let sp = sch_pads(nl, c);
                    let used: BTreeSet<&String> = f
                        .pad_nets
                        .keys()
                        .chain(sp.iter().filter(|(_, n)| n.is_some()).map(|(p, _)| p))
                        .collect();
                    let missing: Vec<&str> = used
                        .iter()
                        .filter(|pad| p.pad(pad).is_none())
                        .map(|s| s.as_str())
                        .collect();
                    if !missing.is_empty() {
                        ch.issues.push((
                            Status::Error,
                            format!(
                                "pads don't match: {} has {} pads, {} has {}; no pad {} for its nets",
                                f.package.name,
                                f.package.pads.len(),
                                p.name,
                                p.pads.len(),
                                missing.join(", ")
                            ),
                        ));
                    } else if p.pads.len() != f.package.pads.len() {
                        ch.issues.push((
                            Status::Warning,
                            format!(
                                "pad count changes from {} to {}",
                                f.package.pads.len(),
                                p.pads.len()
                            ),
                        ));
                    }
                }
            }
            if f.locked {
                ch.issues.push((
                    Status::Warning,
                    format!("{} is locked; its position is kept", f.reference),
                ));
            }
            if board_edit {
                ch.issues.push((
                    Status::Warning,
                    format!(
                        "the footprint was changed on the board; back-annotate to keep {}",
                        f.package.name
                    ),
                ));
            }
            if both {
                ch.conflict = Some(format!(
                    "footprint changed on both sides since the last sync ({to} and {})",
                    f.package.name
                ));
            }
        }
        // Value and parameters.
        let value = component_value(c).unwrap_or_default();
        if !o.ignore_values && value != f.value {
            b.push(
                ChangeKind::ChangeValue,
                &key,
                c.designator.clone(),
                Some(f.value.clone()).filter(|v| !v.is_empty()),
                Some(value.clone()).filter(|v| !v.is_empty()),
                comp_cause(
                    c,
                    root,
                    format!(
                        "{} is {} in the schematic",
                        c.designator,
                        if value.is_empty() { "unvalued" } else { &value }
                    ),
                ),
                Impact {
                    footprints: vec![f.reference.clone()],
                    ..Impact::default()
                },
                Op::SetValue {
                    reference: f.reference.clone(),
                    value,
                },
            );
        }
        if !o.ignore_parameters {
            let want: BTreeMap<String, String> = c
                .params
                .iter()
                .filter(|(k, _)| !o.ignored_parameters.contains(*k))
                .map(|(k, v)| (k.clone(), param_display(v)))
                .collect();
            let mut keys: BTreeSet<String> = want.keys().cloned().collect();
            // Parameters the ECO wrote before and the schematic dropped.
            keys.extend(
                f.properties
                    .keys()
                    .filter(|k| k.starts_with("param:"))
                    .map(|k| k["param:".len()..].to_owned())
                    .filter(|k| !o.ignored_parameters.contains(k)),
            );
            for k in keys {
                let have = f.properties.get(&format!("param:{k}")).cloned();
                let to = want.get(&k).cloned();
                if have == to {
                    continue;
                }
                b.push(
                    ChangeKind::ChangeParameter,
                    &format!("{key}#{k}"),
                    format!("{} {}", c.designator, k.replace('_', " ")),
                    have,
                    to.clone(),
                    comp_cause(
                        c,
                        root,
                        match &to {
                            Some(v) => format!(
                                "{} {} is {v} in the schematic",
                                c.designator,
                                k.replace('_', " ")
                            ),
                            None => {
                                format!("{} no longer sets {}", c.designator, k.replace('_', " "))
                            }
                        },
                    ),
                    Impact {
                        footprints: vec![f.reference.clone()],
                        ..Impact::default()
                    },
                    Op::SetParam {
                        reference: f.reference.clone(),
                        key: k.clone(),
                        value: to,
                    },
                );
            }
        }
        if !o.ignore_fitting && c.fitted() != f.fitted {
            let fit = |x: bool| if x { "fitted" } else { "not fitted" }.to_owned();
            b.push(
                ChangeKind::ChangeFitted,
                &key,
                c.designator.clone(),
                Some(fit(f.fitted)),
                Some(fit(c.fitted())),
                comp_cause(
                    c,
                    root,
                    format!(
                        "{} is {} in variant {}",
                        c.designator,
                        fit(c.fitted()),
                        nl.variant.as_deref().unwrap_or("default")
                    ),
                ),
                Impact {
                    footprints: vec![f.reference.clone()],
                    ..Impact::default()
                },
                Op::SetFitted {
                    reference: f.reference.clone(),
                    fitted: c.fitted(),
                },
            );
        }
        // Pin connections.
        let want = sch_pads(nl, c);
        let mut pads: BTreeSet<String> = want.keys().cloned().collect();
        pads.extend(f.pad_nets.keys().cloned());
        for pad in pads {
            let have_board = f.pad_nets.get(&pad).map(|n| board.net_name(*n).to_owned());
            let have = have_board.as_deref().map(board_as_sch);
            let to = want.get(&pad).cloned().flatten();
            if have == to {
                continue;
            }
            if f.package.pad(&pad).is_none() && have.is_none() {
                // The footprint change reports pads the board footprint doesn't have.
                continue;
            }
            let synced_net = synced.and_then(|s| s.pads.get(&pad)).cloned();
            let board_edit = synced.is_some() && synced_net == to;
            let both =
                synced.is_some() && synced_net != to && synced_net.as_ref() != have_board.as_ref();
            let impact = copper_at_pad(board, *fid, &pad);
            let mut requires = Vec::new();
            if let Some(r) = to.as_deref().and_then(add_net_req) {
                requires.push(r);
            }
            let ch = b.push(
                ChangeKind::ChangePinNet,
                &format!("{key}#{pad}"),
                format!("{}.{}", c.designator, pad),
                have_board.as_deref().map(net_label),
                to.as_deref().map(net_label),
                comp_cause(
                    c,
                    root,
                    match &to {
                        Some(n) => {
                            format!(
                                "pin {}.{} is on {} in the schematic",
                                c.designator,
                                pad,
                                net_label(n)
                            )
                        }
                        None => format!(
                            "pin {}.{} is unconnected in the schematic",
                            c.designator, pad
                        ),
                    },
                ),
                impact,
                Op::SetPadNet {
                    reference: f.reference.clone(),
                    pad: pad.clone(),
                    net: to,
                },
            );
            ch.requires = requires;
            if ch.impact.copper() > 0 {
                ch.issues.push((
                    Status::Warning,
                    format!(
                        "{} routed items on {} touch this pad",
                        ch.impact.copper(),
                        have_board
                            .as_deref()
                            .map(net_label)
                            .unwrap_or_else(|| "no net".into())
                    ),
                ));
            }
            if board_edit {
                ch.issues.push((
                    Status::Warning,
                    "the pin was swapped on the board; back-annotate to keep the swap".to_owned(),
                ));
            }
            if both {
                ch.conflict =
                    Some("the pin's net changed on both sides since the last sync".to_owned());
            }
        }
    }
    new_rooms
}
