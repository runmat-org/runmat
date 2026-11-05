//! The forward compare: what the board needs to match the schematic.

use super::*;
use crate::geom::poly::{rect_ring, Polygon};
use crate::geom::{mm, Coord, Rect};
use crate::ids::NamePath;
use crate::index::{component_value, param_display};
use crate::layout::{Board, ItemId};
use std::collections::{BTreeMap, BTreeSet};

/// Schematic → board: everything the board would change to match the schematic.
pub fn compare(s: &Schematic, board: &Board, o: &ComparatorOptions) -> EcoPlan {
    let nl = s.netlist;
    let root = s.design.root.clone();
    let links = link(s, board, o);
    let mut b = Builder {
        changes: Vec::new(),
        direction: Direction::Forward,
    };
    let base = &board.sync;

    // ---- Nets: renames first (pins kept under a new name keep their routing).
    let sch_names: BTreeSet<&str> = nl.nets.iter().map(|n| n.name.as_str()).collect();
    let board_names: BTreeSet<&str> = board.nets.iter().map(|n| n.name.as_str()).collect();
    // Pin identity: (component path, pad).
    let comp_of_fp: BTreeMap<ItemId, usize> = links.by_comp.iter().map(|(c, f)| (*f, *c)).collect();
    let mut board_pins: BTreeMap<&str, BTreeSet<(String, String)>> = BTreeMap::new();
    for f in board.footprints.values() {
        let Some(ci) = comp_of_fp.get(&f.id) else {
            continue;
        };
        for (pad, net) in &f.pad_nets {
            board_pins
                .entry(board.net_name(*net))
                .or_default()
                .insert((nl.components[*ci].path.to_string(), pad.clone()));
        }
    }
    let mut sch_pins: BTreeMap<&str, BTreeSet<(String, String)>> = BTreeMap::new();
    for n in &nl.nets {
        let set = sch_pins.entry(n.name.as_str()).or_default();
        for (ci, pi) in &n.pins {
            let c = &nl.components[*ci];
            for pad in &c.pins[*pi].pads {
                set.insert((c.path.to_string(), pad.clone()));
            }
        }
    }
    let new_nets: Vec<&str> = sch_names.difference(&board_names).copied().collect();
    let gone_nets: Vec<&str> = board_names.difference(&sch_names).copied().collect();
    // board name → schematic name
    let mut renamed: BTreeMap<String, String> = BTreeMap::new();
    let mut taken: BTreeSet<&str> = BTreeSet::new();
    for old in &gone_nets {
        let Some(bp) = board_pins.get(old).filter(|p| !p.is_empty()) else {
            continue;
        };
        let mut best: Option<(&str, f64)> = None;
        for new in &new_nets {
            if taken.contains(new) {
                continue;
            }
            let explicit = nl
                .net(new)
                .and_then(|(_, n)| n.renamed_from.as_ref())
                .is_some_and(|r| r.leaf() == NamePath::parse(old).leaf());
            let sp = sch_pins.get(new).cloned().unwrap_or_default();
            let inter = bp.intersection(&sp).count() as f64;
            let uni = bp.union(&sp).count() as f64;
            let score = if explicit {
                2.0
            } else if uni > 0.0 {
                inter / uni
            } else {
                0.0
            };
            if score >= 0.5 && best.is_none_or(|(_, s)| score > s) {
                best = Some((new, score));
            }
        }
        if let Some((new, _)) = best {
            taken.insert(new);
            renamed.insert((*old).to_owned(), new.to_owned());
        }
    }
    for (old, new) in &renamed {
        let nid = board.net(old).unwrap();
        let (_, n) = nl.net(new).unwrap();
        let (sheet, sheet_path) = sheet_of(&n.path, &root);
        let mut impact = copper_on_net(board, nid);
        impact.nets.push(new.clone());
        b.push(
            ChangeKind::RenameNet,
            old,
            net_label(new),
            Some(net_label(old)),
            Some(net_label(new)),
            Cause {
                sheet,
                sheet_path,
                source: None,
                reason: if n.renamed_from.is_some() {
                    format!("net renamed from {old} in the schematic")
                } else {
                    format!("the pins of {old} are now named {new}")
                },
            },
            impact,
            Op::RenameNet {
                from: old.clone(),
                to: new.clone(),
            },
        );
    }
    // A board net name as the schematic now calls it.
    let board_as_sch = |name: &str| -> String {
        renamed
            .get(name)
            .cloned()
            .unwrap_or_else(|| name.to_owned())
    };
    let renamed_to: BTreeSet<&str> = renamed.values().map(|s| s.as_str()).collect();
    for new in &new_nets {
        if renamed_to.contains(new) {
            continue;
        }
        let (_, n) = nl.net(new).unwrap();
        if n.pins.is_empty() {
            continue;
        }
        let (sheet, sheet_path) = sheet_of(&n.path, &root);
        b.push(
            ChangeKind::AddNet,
            new,
            net_label(new),
            None,
            n.class.as_ref().map(|c| format!("class {c}")),
            Cause {
                sheet,
                sheet_path,
                source: None,
                reason: format!(
                    "{} has {} pins in the schematic",
                    net_label(new),
                    n.pins.len()
                ),
            },
            Impact {
                nets: vec![(*new).to_owned()],
                ..Impact::default()
            },
            Op::AddNet {
                name: (*new).to_owned(),
                class: n.class.clone(),
            },
        );
    }
    let add_net_req = |name: &str| -> Option<String> {
        new_nets
            .contains(&name)
            .then(|| {
                if renamed_to.contains(name) {
                    renamed
                        .iter()
                        .find(|(_, v)| v.as_str() == name)
                        .map(|(k, _)| format!("rename_net:{k}"))
                } else {
                    Some(format!("add_net:{name}"))
                }
            })
            .flatten()
    };
    for old in &gone_nets {
        if renamed.contains_key(*old) {
            continue;
        }
        let nid = board.net(old).unwrap();
        let impact = copper_on_net(board, nid);
        let ch = b.push(
            ChangeKind::RemoveNet,
            old,
            net_label(old),
            Some(net_label(old)),
            None,
            root_cause(
                &root,
                None,
                format!("{} is no longer in the schematic", net_label(old)),
            ),
            impact,
            Op::RemoveNet {
                name: (*old).to_owned(),
            },
        );
        if ch.impact.copper() > 0 {
            let n = ch.impact.copper();
            ch.issues.push((
                Status::Warning,
                format!("{n} routed items on {} lose their net", net_label(old)),
            ));
        }
    }

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
                    &root,
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
                &root,
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
                    &root,
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
                comp_cause(c, &root, format!("the part of {} uses {to}", c.designator)),
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
                    &root,
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
                        &root,
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
                    &root,
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
                    &root,
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

    // ---- Net classes, membership, rules.
    if !o.ignore_net_classes {
        for (name, nc) in &s.design.net_classes {
            let params: BTreeMap<String, String> = nc
                .params
                .iter()
                .map(|(k, v)| (k.clone(), param_display(v)))
                .collect();
            let src = (!nc.source.file.is_empty()).then(|| nc.source.to_string());
            match base.net_classes.get(name) {
                None => {
                    b.push(
                        ChangeKind::AddNetClass,
                        name,
                        name.clone(),
                        None,
                        Some(format!("{} rules", params.len())),
                        root_cause(
                            &root,
                            src.clone(),
                            format!("net class {name} is declared in the schematic"),
                        ),
                        Impact::default(),
                        Op::SetNetClassDef {
                            name: name.clone(),
                            params: Some(params.clone()),
                        },
                    );
                }
                Some(have) if !o.ignore_rules => {
                    let keys: BTreeSet<&String> = have.keys().chain(params.keys()).collect();
                    for k in keys {
                        let (from, to) = (have.get(k).cloned(), params.get(k).cloned());
                        if from == to {
                            continue;
                        }
                        let nets: Vec<String> = board
                            .nets
                            .iter()
                            .filter(|n| n.class.as_deref() == Some(name))
                            .map(|n| n.name.clone())
                            .collect();
                        b.push(
                            ChangeKind::ChangeRule,
                            &format!("{name}#{k}"),
                            format!("{name} {}", k.replace('_', " ")),
                            from,
                            to,
                            root_cause(
                                &root,
                                src.clone(),
                                format!(
                                    "net class {name} sets {} in the schematic",
                                    k.replace('_', " ")
                                ),
                            ),
                            Impact {
                                nets,
                                ..Impact::default()
                            },
                            Op::SetRule {
                                class: name.clone(),
                                key: k.clone(),
                                value: params.get(k).cloned(),
                            },
                        );
                    }
                }
                _ => {}
            }
        }
        for name in base
            .net_classes
            .keys()
            .filter(|k| !s.design.net_classes.contains_key(*k))
        {
            b.push(
                ChangeKind::RemoveNetClass,
                name,
                name.clone(),
                Some(name.clone()),
                None,
                root_cause(
                    &root,
                    None,
                    format!("net class {name} is no longer declared"),
                ),
                Impact::default(),
                Op::SetNetClassDef {
                    name: name.clone(),
                    params: None,
                },
            );
        }
        for n in &nl.nets {
            let Some(bn) = board.nets.iter().find(|x| board_as_sch(&x.name) == n.name) else {
                continue;
            };
            if bn.class == n.class {
                continue;
            }
            let (sheet, sheet_path) = sheet_of(&n.path, &root);
            let mut requires = Vec::new();
            if let Some(cls) = &n.class {
                if !base.net_classes.contains_key(cls) {
                    requires.push(format!("add_net_class:{cls}"));
                }
            }
            if let Some(r) = add_net_req(&n.name) {
                requires.push(r);
            }
            let ch = b.push(
                ChangeKind::ChangeNetClass,
                &n.name,
                net_label(&n.name),
                bn.class.clone(),
                n.class.clone(),
                Cause {
                    sheet,
                    sheet_path,
                    source: None,
                    reason: match &n.class {
                        Some(c) => format!("{} is in net class {c}", net_label(&n.name)),
                        None => format!("{} has no net class", net_label(&n.name)),
                    },
                },
                Impact {
                    nets: vec![n.name.clone()],
                    ..Impact::default()
                },
                Op::SetNetClass {
                    net: n.name.clone(),
                    class: n.class.clone(),
                },
            );
            ch.requires = requires;
        }
    }
    // Nets added with a class carry it; their class must exist.
    for ch in b.changes.iter_mut() {
        if let Op::AddNet {
            class: Some(cls), ..
        } = &ch.op
        {
            if !o.ignore_net_classes
                && !base.net_classes.contains_key(cls)
                && s.design.net_classes.contains_key(cls)
            {
                ch.requires.push(format!("add_net_class:{cls}"));
            }
        }
    }

    // ---- Differential pairs.
    if !o.ignore_diff_pairs {
        let mut want: BTreeMap<String, (String, String)> = BTreeMap::new();
        for d in &nl.diff_pairs {
            want.insert(
                d.name.clone(),
                (nl.nets[d.p].name.clone(), nl.nets[d.n].name.clone()),
            );
        }
        for (name, pair) in &want {
            if base.diff_pairs.get(name) == Some(pair) {
                continue;
            }
            let mut requires: Vec<String> = [&pair.0, &pair.1]
                .iter()
                .filter_map(|n| add_net_req(n))
                .collect();
            requires.sort();
            let ch = b.push(
                ChangeKind::AddDiffPair,
                name,
                name.clone(),
                base.diff_pairs.get(name).map(|(p, n)| format!("{p} / {n}")),
                Some(format!("{} / {}", pair.0, pair.1)),
                root_cause(
                    &root,
                    None,
                    format!("differential pair {name} is declared in the schematic"),
                ),
                Impact {
                    nets: vec![pair.0.clone(), pair.1.clone()],
                    ..Impact::default()
                },
                Op::SetDiffPair {
                    name: name.clone(),
                    pair: Some(pair.clone()),
                },
            );
            ch.requires = requires;
        }
        for (name, pair) in base
            .diff_pairs
            .iter()
            .filter(|(k, _)| !want.contains_key(*k))
        {
            b.push(
                ChangeKind::RemoveDiffPair,
                name,
                name.clone(),
                Some(format!("{} / {}", pair.0, pair.1)),
                None,
                root_cause(
                    &root,
                    None,
                    format!("differential pair {name} is no longer declared"),
                ),
                Impact {
                    nets: vec![pair.0.clone(), pair.1.clone()],
                    ..Impact::default()
                },
                Op::SetDiffPair {
                    name: name.clone(),
                    pair: None,
                },
            );
        }
    }

    // ---- Component classes and rooms (multi-channel).
    let mut channels: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    let mut authored: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for (ci, c) in nl.components.iter().enumerate() {
        if let Some(ch) = channel_of(c) {
            channels.entry(ch).or_default().push(ci);
        }
        for cls in &c.classes {
            authored.entry(cls.clone()).or_default().push(ci);
        }
    }
    let designators = |cis: &[usize]| -> Vec<String> {
        let mut v: Vec<String> = cis
            .iter()
            .map(|ci| nl.components[*ci].designator.clone())
            .collect();
        v.sort_by_key(|d| natural(d));
        v
    };
    let add_comp_req = |cis: &[usize]| -> Vec<String> {
        cis.iter()
            .filter(|ci| !links.by_comp.contains_key(ci))
            .map(|ci| format!("add_component:{}", nl.components[*ci].path))
            .collect()
    };
    let first_comp = |cis: &[usize], why: String| -> Cause {
        cis.first()
            .map(|ci| {
                let c = &nl.components[*ci];
                let (sheet, sheet_path) = sheet_of(&c.path, &root);
                Cause {
                    sheet,
                    sheet_path,
                    source: None,
                    reason: why.clone(),
                }
            })
            .unwrap_or_else(|| root_cause(&root, None, why))
    };
    if !o.ignore_component_classes {
        let mut want: BTreeMap<String, Vec<usize>> = authored.clone();
        if o.classes_per_channel {
            for (k, v) in &channels {
                want.insert(k.clone(), v.clone());
            }
        }
        for (name, cis) in &want {
            let members = designators(cis);
            let have = base.component_classes.get(name);
            if have == Some(&members) {
                continue;
            }
            let kind = if have.is_some() {
                ChangeKind::ChangeComponentClass
            } else {
                ChangeKind::AddComponentClass
            };
            let ch = b.push(
                kind,
                name,
                name.clone(),
                have.map(|m| format!("{} members", m.len())),
                Some(format!("{} members", members.len())),
                first_comp(
                    cis,
                    if channels.contains_key(name) {
                        format!("channel {name} of a repeated sheet")
                    } else {
                        format!("components tagged {name}")
                    },
                ),
                Impact {
                    footprints: members.clone(),
                    ..Impact::default()
                },
                Op::SetComponentClass {
                    name: name.clone(),
                    members: Some(members),
                },
            );
            ch.requires = add_comp_req(cis);
        }
        for name in base
            .component_classes
            .keys()
            .filter(|k| !want.contains_key(*k))
        {
            b.push(
                ChangeKind::RemoveComponentClass,
                name,
                name.clone(),
                Some(name.clone()),
                None,
                root_cause(
                    &root,
                    None,
                    format!("component class {name} is gone from the schematic"),
                ),
                Impact::default(),
                Op::SetComponentClass {
                    name: name.clone(),
                    members: None,
                },
            );
        }
    }
    if !o.ignore_rooms && o.rooms_per_channel {
        let fp_ref_of = |ci: usize| links.by_comp.get(&ci).map(|f| board.footprints[f].bbox());
        for (name, cis) in &channels {
            let members = designators(cis);
            let existing = board.rooms.iter().find(|r| &r.name == name);
            match existing {
                Some(r) => {
                    let mut have = r.members.clone();
                    have.sort_by_key(|d| natural(d));
                    if have == members {
                        continue;
                    }
                    let ch = b.push(
                        ChangeKind::ChangeRoom,
                        name,
                        name.clone(),
                        Some(format!("{} parts", have.len())),
                        Some(format!("{} parts", members.len())),
                        first_comp(cis, format!("channel {name} has {} parts", members.len())),
                        Impact {
                            footprints: members.clone(),
                            ..Impact::default()
                        },
                        Op::SetRoom {
                            name: name.clone(),
                            outline: None,
                            members,
                        },
                    );
                    ch.requires = add_comp_req(cis);
                }
                None => {
                    // Around the channel's placed parts, else its staging block.
                    let placed = cis
                        .iter()
                        .filter_map(|ci| fp_ref_of(*ci))
                        .fold(Rect::EMPTY, |a, r| a.union(r));
                    let area = if placed.is_empty() {
                        new_rooms.get(name).copied().unwrap_or(Rect::EMPTY)
                    } else {
                        placed.inflate(mm(1.0))
                    };
                    if area.is_empty() {
                        continue;
                    }
                    let staged = placed.is_empty();
                    let ch = b.push(
                        ChangeKind::AddRoom,
                        name,
                        name.clone(),
                        None,
                        Some(format!("{} parts", members.len())),
                        first_comp(
                            cis,
                            format!("channel {name} of a repeated sheet gets a room"),
                        ),
                        Impact {
                            footprints: members.clone(),
                            placement: Some(Placement {
                                at: area.center(),
                                room: Some(name.clone()),
                                staged,
                            }),
                            ..Impact::default()
                        },
                        Op::SetRoom {
                            name: name.clone(),
                            outline: Some(Polygon::from_ring(rect_ring(area))),
                            members,
                        },
                    );
                    ch.requires = add_comp_req(cis);
                }
            }
        }
        for r in board
            .rooms
            .iter()
            .filter(|r| r.name.ends_with(']') && !channels.contains_key(&r.name))
        {
            b.push(
                ChangeKind::RemoveRoom,
                &r.name,
                r.name.clone(),
                Some(format!("{} parts", r.members.len())),
                None,
                root_cause(
                    &root,
                    None,
                    format!("channel {} is no longer in the schematic", r.name),
                ),
                Impact::default(),
                Op::SetRoom {
                    name: r.name.clone(),
                    outline: None,
                    members: Vec::new(),
                },
            );
        }
    }

    finish(Direction::Forward, b.changes, o)
}
