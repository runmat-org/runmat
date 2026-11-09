//! Net classes, their membership and rules, and differential pairs.

use super::*;
use crate::index::param_display;
use std::collections::{BTreeMap, BTreeSet};

pub(super) fn compare_classes(x: &Inputs, nets: &Nets, b: &mut Builder) {
    let Inputs {
        s,
        nl,
        board,
        o,
        root,
        base,
        ..
    } = *x;
    let board_as_sch = |name: &str| nets.board_as_sch(name);
    let add_net_req = |name: &str| nets.add_req(name);
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
}

pub(super) fn compare_pairs(x: &Inputs, nets: &Nets, b: &mut Builder) {
    let Inputs {
        nl, o, root, base, ..
    } = *x;
    let add_net_req = |name: &str| nets.add_req(name);
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
}
