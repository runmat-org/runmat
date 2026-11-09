//! Component classes and rooms: one class and one room per channel (multi-channel), and the
//! classes the schematic authors.

use super::*;
use crate::geom::poly::{rect_ring, Polygon};
use crate::geom::{mm, Rect};
use std::collections::BTreeMap;

pub(super) fn compare_channels(x: &Inputs, new_rooms: &BTreeMap<String, Rect>, b: &mut Builder) {
    let Inputs {
        nl,
        board,
        o,
        root,
        links,
        base,
        ..
    } = *x;
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
                let (sheet, sheet_path) = sheet_of(&c.path, root);
                Cause {
                    sheet,
                    sheet_path,
                    source: None,
                    reason: why.clone(),
                }
            })
            .unwrap_or_else(|| root_cause(root, None, why))
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
                    root,
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
                    root,
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
}
