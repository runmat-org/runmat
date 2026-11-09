//! Nets: renames first (pins kept under a new name keep their routing), then added and
//! removed nets.

use super::*;
use crate::ids::NamePath;
use crate::layout::ItemId;
use std::collections::{BTreeMap, BTreeSet};

pub(super) fn compare_nets(x: &Inputs, b: &mut Builder) -> Nets {
    let Inputs {
        nl,
        board,
        root,
        links,
        ..
    } = *x;
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
    Nets {
        renamed,
        new: new_nets.iter().map(|n| (*n).to_owned()).collect(),
    }
}
