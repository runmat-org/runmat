//! Validating a selection of lines, applying them to the board, undo and the checks after.

use super::*;
use crate::board_rules::BItem;
use crate::board_rules::BoardRules;
use crate::connectivity::{islands, net_nodes};
use crate::drc::Drc;
use crate::elab::{Diagnostic, Severity};
use crate::layout::{Board, ItemId, NetId, Room, Side};
use crate::library::Pool;
use std::collections::{BTreeMap, BTreeSet};

/* ------------------------------------------------------------------------------------------------
 * Validation
 * --------------------------------------------------------------------------------------------- */

#[derive(Clone, Debug, PartialEq)]
pub struct Check {
    pub status: Status,
    pub reasons: Vec<String>,
}

impl Check {
    pub fn reason(&self) -> Option<String> {
        (!self.reasons.is_empty()).then(|| self.reasons.join("; "))
    }
}

/// Status of each selected change, given the whole selection.
pub fn validate(
    plan: &EcoPlan,
    board: &Board,
    selected: &BTreeSet<String>,
) -> BTreeMap<String, Check> {
    // Designators the selection frees and claims.
    let mut freed: BTreeSet<&str> = BTreeSet::new();
    let mut claimed: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
    for c in plan.changes.iter().filter(|c| selected.contains(&c.id)) {
        match &c.op {
            Op::RemoveComponent { reference } => {
                freed.insert(reference);
            }
            Op::SetDesignator { reference, to, .. } => {
                freed.insert(reference);
                claimed.entry(to).or_default().push(&c.id);
            }
            Op::AddComponent { designator, .. } => {
                claimed.entry(designator).or_default().push(&c.id)
            }
            _ => {}
        }
    }
    let mut out = BTreeMap::new();
    for c in plan.changes.iter().filter(|c| selected.contains(&c.id)) {
        let mut status = Status::Ok;
        let mut reasons = Vec::new();
        let mut raise = |s: Status, r: String, status: &mut Status| {
            *status = (*status).max(s);
            reasons.push(r);
        };
        if let Some(conflict) = &c.conflict {
            raise(Status::Error, format!("conflict: {conflict}"), &mut status);
        }
        for (s, r) in &c.issues {
            raise(*s, r.clone(), &mut status);
        }
        for r in &c.requires {
            if !selected.contains(r) {
                let what = plan
                    .get(r)
                    .map(|x| format!("{} {}", x.kind.label().to_lowercase(), x.item))
                    .unwrap_or_else(|| r.clone());
                raise(
                    Status::Error,
                    format!("needs {what} (not selected)"),
                    &mut status,
                );
            }
        }
        let claim = match &c.op {
            Op::SetDesignator { to, .. } => Some(to.as_str()),
            Op::AddComponent { designator, .. } => Some(designator.as_str()),
            _ => None,
        };
        if let Some(d) = claim {
            if claimed.get(d).is_some_and(|v| v.len() > 1) {
                raise(
                    Status::Error,
                    format!("{d} is claimed by more than one change"),
                    &mut status,
                );
            } else if board.footprint_by_ref(d).is_some() && !freed.contains(d) {
                raise(
                    Status::Error,
                    format!("{d} is already used on the board"),
                    &mut status,
                );
            }
        }
        out.insert(c.id.clone(), Check { status, reasons });
    }
    out
}

/* ------------------------------------------------------------------------------------------------
 * Apply
 * --------------------------------------------------------------------------------------------- */

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ApplyOptions {
    /// Delete the tracks and vias a change leaves without a net or pad (else they keep no net).
    pub remove_orphaned_copper: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ApplyReport {
    pub applied: Vec<String>,
    pub removed_tracks: Vec<u64>,
    pub removed_vias: Vec<u64>,
    /// Copper left on no net (kept because orphan removal is off).
    pub unnetted: Vec<u64>,
    /// The board before the edit: the one undo step.
    pub undo: Box<Board>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ApplyError {
    /// Each selected change that failed validation, with why.
    pub failures: Vec<(String, String)>,
}

impl std::fmt::Display for ApplyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} changes failed validation", self.failures.len())
    }
}

pub(super) fn net_or_add(board: &mut Board, name: &str) -> NetId {
    board.add_net(name, None)
}

/// Executes the selected forward changes as one edit: validated first, all or nothing.
/// Placement and routing nothing touches stay as they were.
pub fn apply(
    board: &mut Board,
    plan: &EcoPlan,
    selected: &BTreeSet<String>,
    pool: &Pool,
    o: &ApplyOptions,
) -> Result<ApplyReport, ApplyError> {
    let checks = validate(plan, board, selected);
    let failures: Vec<(String, String)> = checks
        .iter()
        .filter(|(_, c)| c.status == Status::Error)
        .map(|(id, c)| (id.clone(), c.reason().unwrap_or_default()))
        .collect();
    if !failures.is_empty() {
        return Err(ApplyError { failures });
    }
    let before = board.clone();
    let mut next = board.clone();
    // The settings this update was compared with are the board's from now on.
    next.sync.options = plan.options.clone();
    let mut changes: Vec<&Change> = plan
        .changes
        .iter()
        .filter(|c| selected.contains(&c.id) && c.direction == Direction::Forward)
        .collect();
    // Net renames run first: an added part's pads look their nets up by the new names, and
    // would otherwise create a second net under a name a rename is about to take.
    changes.sort_by_key(|c| (c.kind != ChangeKind::RenameNet, c.kind));
    let mut removed_nets: BTreeSet<NetId> = BTreeSet::new();
    let mut touched_nets: BTreeSet<NetId> = BTreeSet::new();
    let mut moved_pads: Vec<(String, String)> = Vec::new();
    // Designators re-designated earlier in this apply: later lines name the part by its old one.
    let mut renamed: BTreeMap<String, String> = BTreeMap::new();
    let by_ref = |b: &Board, r: &str| {
        b.footprints
            .values()
            .find(|f| f.reference == r)
            .map(|f| f.id)
    };
    for c in &changes {
        match &c.op {
            Op::RemoveComponent { reference } => {
                if let Some(id) = by_ref(&next, renamed.get(reference).unwrap_or(reference)) {
                    let f = next.footprints.remove(&id).unwrap();
                    touched_nets.extend(f.pad_nets.values().copied());
                    if let Some(p) = &f.component {
                        next.sync.components.remove(p);
                    }
                    // Ignore rooms leaves every room as it is, members included.
                    if !plan.options.ignore_rooms {
                        for r in next.rooms.iter_mut() {
                            r.members.retain(|m| m != reference);
                        }
                    }
                }
            }
            Op::AddComponent {
                path,
                designator,
                package,
                value,
                params,
                pads,
                fitted,
                at,
            } => {
                let Some(pkg) = package.and_then(|u| pool.packages.get(&u)) else {
                    continue;
                };
                let id = next.place(designator, pkg.clone(), *at, 0.0, Side::Top);
                let mut nets = BTreeMap::new();
                for (pad, n) in pads {
                    nets.insert(pad.clone(), net_or_add(&mut next, n));
                }
                let f = next.footprints.get_mut(&id).unwrap();
                f.component = Some(path.clone());
                f.value = value.clone();
                f.fitted = *fitted;
                f.pad_nets = nets;
                for (k, v) in params {
                    f.properties.insert(format!("param:{k}"), v.clone());
                }
                next.sync.components.insert(
                    path.clone(),
                    SyncedComponent {
                        designator: designator.clone(),
                        package: pkg.name.clone(),
                        pads: pads.clone(),
                    },
                );
            }
            Op::SetDesignator {
                reference,
                to,
                path,
            } => {
                if let Some(id) = by_ref(&next, renamed.get(reference).unwrap_or(reference)) {
                    let f = next.footprints.get_mut(&id).unwrap();
                    f.reference = to.clone();
                    renamed.insert(reference.clone(), to.clone());
                    let old_link = f.component.replace(path.clone());
                    let mut rec = old_link
                        .and_then(|p| next.sync.components.remove(&p))
                        .or_else(|| next.sync.components.remove(path));
                    if let Some(r) = rec.as_mut() {
                        r.designator = to.clone();
                    }
                    if let Some(r) = rec {
                        next.sync.components.insert(path.clone(), r);
                    }
                    for r in next.rooms.iter_mut() {
                        for m in r.members.iter_mut().filter(|m| *m == reference) {
                            *m = to.clone();
                        }
                    }
                }
            }
            Op::SetFootprint { reference, package } => {
                let Some(pkg) = package.and_then(|u| pool.packages.get(&u)) else {
                    continue;
                };
                if let Some(id) = by_ref(&next, renamed.get(reference).unwrap_or(reference)) {
                    let f = next.footprints.get_mut(&id).unwrap();
                    f.package = pkg.clone();
                    f.pad_nets.retain(|pad, _| pkg.pad(pad).is_some());
                    if let Some(s) = f
                        .component
                        .clone()
                        .and_then(|p| next.sync.components.get_mut(&p))
                    {
                        s.package = pkg.name.clone();
                    }
                }
            }
            Op::SetValue { reference, value } => {
                if let Some(id) = by_ref(&next, renamed.get(reference).unwrap_or(reference)) {
                    next.footprints.get_mut(&id).unwrap().value = value.clone();
                }
            }
            Op::SetParam {
                reference,
                key,
                value,
            } => {
                if let Some(id) = by_ref(&next, renamed.get(reference).unwrap_or(reference)) {
                    let props = &mut next.footprints.get_mut(&id).unwrap().properties;
                    match value {
                        Some(v) => props.insert(format!("param:{key}"), v.clone()),
                        None => props.remove(&format!("param:{key}")),
                    };
                }
            }
            Op::SetFitted { reference, fitted } => {
                if let Some(id) = by_ref(&next, renamed.get(reference).unwrap_or(reference)) {
                    next.footprints.get_mut(&id).unwrap().fitted = *fitted;
                }
            }
            Op::RenameNet { from, to } => {
                if let Some(n) = next.net(from) {
                    next.nets[n.0 as usize].name = to.clone();
                    for s in next.sync.components.values_mut() {
                        for v in s.pads.values_mut().filter(|v| *v == from) {
                            *v = to.clone();
                        }
                    }
                }
            }
            Op::AddNet { name, class } => {
                let n = next.add_net(name, class.as_deref());
                next.nets[n.0 as usize].class = class.clone();
            }
            Op::SetPadNet {
                reference,
                pad,
                net,
            } => {
                if let Some(id) = by_ref(&next, renamed.get(reference).unwrap_or(reference)) {
                    let new = net.as_deref().map(|n| net_or_add(&mut next, n));
                    let f = next.footprints.get_mut(&id).unwrap();
                    let old = match new {
                        Some(n) => f.pad_nets.insert(pad.clone(), n),
                        None => f.pad_nets.remove(pad),
                    };
                    touched_nets.extend(old);
                    moved_pads.push((reference.clone(), pad.clone()));
                    let path = f.component.clone();
                    if let Some(s) = path.and_then(|p| next.sync.components.get_mut(&p)) {
                        match net {
                            Some(n) => s.pads.insert(pad.clone(), n.clone()),
                            None => s.pads.remove(pad),
                        };
                    }
                }
            }
            Op::RemoveNet { name } => {
                if let Some(n) = next.net(name) {
                    removed_nets.insert(n);
                }
            }
            Op::SetNetClassDef { name, params } => match params {
                Some(p) => {
                    next.sync.net_classes.insert(name.clone(), p.clone());
                }
                None => {
                    next.sync.net_classes.remove(name);
                    for n in next
                        .nets
                        .iter_mut()
                        .filter(|n| n.class.as_deref() == Some(name))
                    {
                        n.class = None;
                    }
                }
            },
            Op::SetNetClass { net, class } => {
                if let Some(n) = next.net(net) {
                    next.nets[n.0 as usize].class = class.clone();
                }
            }
            Op::SetDiffPair { name, pair } => match pair {
                Some(p) => {
                    next.sync.diff_pairs.insert(name.clone(), p.clone());
                }
                None => {
                    next.sync.diff_pairs.remove(name);
                }
            },
            Op::SetComponentClass { name, members } => match members {
                Some(m) => {
                    next.sync.component_classes.insert(name.clone(), m.clone());
                }
                None => {
                    next.sync.component_classes.remove(name);
                }
            },
            Op::SetRoom {
                name,
                outline,
                members,
            } => {
                if let Some(r) = next.rooms.iter_mut().find(|r| &r.name == name) {
                    if members.is_empty() && outline.is_none() {
                        next.rooms.retain(|r| &r.name != name);
                    } else {
                        r.members = members.clone();
                        if let Some(o) = outline {
                            r.outline = o.clone();
                        }
                    }
                } else if let Some(o) = outline {
                    next.rooms.push(Room {
                        name: name.clone(),
                        outline: o.clone(),
                        members: members.clone(),
                    });
                    next.rooms
                        .sort_by(|a, b| natural(&a.name).cmp(&natural(&b.name)));
                }
            }
            Op::SetRule { class, key, value } => {
                let e = next.sync.net_classes.entry(class.clone()).or_default();
                match value {
                    Some(v) => e.insert(key.clone(), v.clone()),
                    None => e.remove(key),
                };
            }
            Op::SchDesignator { .. }
            | Op::SchSwapPins { .. }
            | Op::SchSwapGates { .. }
            | Op::SchFootprint { .. } => {}
        }
    }
    // Copper the changes orphaned.
    let mut removed_tracks = Vec::new();
    let mut removed_vias = Vec::new();
    let mut unnetted = Vec::new();
    for n in &removed_nets {
        for t in next.tracks.values_mut().filter(|t| t.net == Some(*n)) {
            t.net = None;
            unnetted.push(t.id.0);
        }
        for v in next.vias.values_mut().filter(|v| v.net == Some(*n)) {
            v.net = None;
            unnetted.push(v.id.0);
        }
        for z in next.zones.values_mut().filter(|z| z.net == Some(*n)) {
            z.net = None;
        }
    }
    if o.remove_orphaned_copper {
        // Unnetted copper from removed nets, plus copper of touched nets left with no pad.
        let unnetted_set: BTreeSet<u64> = unnetted.iter().copied().collect();
        next.tracks.retain(|id, _| {
            let keep = !unnetted_set.contains(&id.0);
            if !keep {
                removed_tracks.push(id.0);
            }
            keep
        });
        next.vias.retain(|id, _| {
            let keep = !unnetted_set.contains(&id.0);
            if !keep {
                removed_vias.push(id.0);
            }
            keep
        });
        unnetted.clear();
        // Copper still on an old net that touches a pad now on another net shorts it.
        for (r, pad) in &moved_pads {
            let Some(fid) = by_ref(&before, r) else {
                continue;
            };
            let imp = copper_at_pad(&before, fid, pad);
            for t in imp.tracks {
                if next.tracks.remove(&ItemId(t)).is_some() {
                    removed_tracks.push(t);
                }
            }
            for v in imp.vias {
                if next.vias.remove(&ItemId(v)).is_some() {
                    removed_vias.push(v);
                }
            }
        }
        let nodes = net_nodes(&next);
        for net in &touched_nets {
            let Some(ns) = nodes.get(net) else { continue };
            let conn = islands(*net, ns.clone());
            let mut with_pad = vec![false; conn.islands];
            for (k, node) in conn.nodes.iter().enumerate() {
                if matches!(node.item, BItem::Pad(..)) {
                    with_pad[conn.island[k]] = true;
                }
            }
            for (k, node) in conn.nodes.iter().enumerate() {
                if with_pad[conn.island[k]] {
                    continue;
                }
                match node.item {
                    BItem::Track(id) => {
                        if next.tracks.remove(&id).is_some() {
                            removed_tracks.push(id.0);
                        }
                    }
                    BItem::Via(id) => {
                        if next.vias.remove(&id).is_some() {
                            removed_vias.push(id.0);
                        }
                    }
                    _ => {}
                }
            }
        }
    }
    if !removed_nets.is_empty() {
        compact_nets(&mut next, &removed_nets);
    }
    removed_tracks.sort();
    removed_tracks.dedup();
    removed_vias.sort();
    removed_vias.dedup();
    unnetted.sort();
    *board = next;
    Ok(ApplyReport {
        applied: changes.iter().map(|c| c.id.clone()).collect(),
        removed_tracks,
        removed_vias,
        unnetted,
        undo: Box::new(before),
    })
}

/// Drops nets and renumbers every net reference (net ids are indices).
pub(super) fn compact_nets(board: &mut Board, removed: &BTreeSet<NetId>) {
    let mut map: BTreeMap<NetId, NetId> = BTreeMap::new();
    let mut nets = Vec::new();
    for (i, n) in board.nets.iter().enumerate() {
        let id = NetId(i as u32);
        if !removed.contains(&id) {
            map.insert(id, NetId(nets.len() as u32));
            nets.push(n.clone());
        }
    }
    board.nets = nets;
    let re = |n: Option<NetId>| n.and_then(|n| map.get(&n).copied());
    for f in board.footprints.values_mut() {
        f.pad_nets = f
            .pad_nets
            .iter()
            .filter_map(|(p, n)| map.get(n).map(|m| (p.clone(), *m)))
            .collect();
    }
    for t in board.tracks.values_mut() {
        t.net = re(t.net);
    }
    for v in board.vias.values_mut() {
        v.net = re(v.net);
    }
    for z in board.zones.values_mut() {
        z.net = re(z.net);
    }
}

/// Reverts an applied ECO (the one undo step).
pub fn undo(board: &mut Board, report: ApplyReport) {
    *board = *report.undo;
}

/// What the checks found after an ECO.
#[derive(Clone, Debug, PartialEq)]
pub struct CheckSummary {
    pub drc_errors: usize,
    pub drc_warnings: usize,
    /// DRC finding messages, sorted.
    pub drc: Vec<String>,
    pub erc_errors: usize,
    pub erc_warnings: usize,
}

/// Applies, refills pours and runs DRC on the board and ERC on the schematic.
pub fn apply_and_check(
    board: &mut Board,
    plan: &EcoPlan,
    selected: &BTreeSet<String>,
    pool: &Pool,
    o: &ApplyOptions,
    rules: &mut BoardRules,
    erc: &[Diagnostic],
) -> Result<(ApplyReport, CheckSummary), ApplyError> {
    let report = apply(board, plan, selected, pool, o)?;
    Ok((report, check(board, rules, erc)))
}

/// DRC (after a pour refill) and ERC counts.
pub fn check(board: &mut Board, rules: &mut BoardRules, erc: &[Diagnostic]) -> CheckSummary {
    let pairs: Vec<(NetId, NetId)> = board
        .sync
        .diff_pairs
        .values()
        .filter_map(|(p, n)| Some((board.net(p)?, board.net(n)?)))
        .collect();
    rules.rebuild(&crate::board_rules::BoardGraph::new(board, &pairs));
    let fills = crate::zone::fill_zones(board, rules, &pairs);
    let drc = Drc::run(board, rules, &pairs, &fills);
    let found = drc.findings();
    let mut msgs: Vec<String> = found.iter().map(|f| f.message.clone()).collect();
    msgs.sort();
    CheckSummary {
        drc_errors: found
            .iter()
            .filter(|f| f.severity == Severity::Error)
            .count(),
        drc_warnings: found
            .iter()
            .filter(|f| f.severity == Severity::Warning)
            .count(),
        drc: msgs,
        erc_errors: erc.iter().filter(|d| d.severity == Severity::Error).count(),
        erc_warnings: erc
            .iter()
            .filter(|d| d.severity == Severity::Warning)
            .count(),
    }
}
