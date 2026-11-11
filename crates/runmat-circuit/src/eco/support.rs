//! Matching schematic parts to footprints, the board impact of a change, placement of new
//! parts and the change-list builder the compares share.

use super::*;
use crate::elab::{CompInst, Netlist};
use crate::geom::shape::Prim;
use crate::geom::{mm, Coord, Point, Rect};
use crate::ids::{LockSection, NamePath, Uuid};
use crate::layout::{Board, ItemId, NetId};
use crate::library::Pool;
use std::collections::{BTreeMap, BTreeSet};

/* ------------------------------------------------------------------------------------------------
 * Matching
 * --------------------------------------------------------------------------------------------- */

pub(super) fn follow(lock: &LockSection, p: &NamePath) -> NamePath {
    let mut cur = p.clone();
    let mut seen = BTreeSet::new();
    while let Some(n) = lock.aliases.get(&cur) {
        if !seen.insert(cur.clone()) {
            break;
        }
        cur = n.clone();
    }
    cur
}

/// Component index → footprint id, and the footprints nothing links to.
pub(super) struct Links {
    pub(super) by_comp: BTreeMap<usize, ItemId>,
    pub(super) unlinked: Vec<ItemId>,
}

pub(super) fn link(s: &Schematic, board: &Board, o: &ComparatorOptions) -> Links {
    let by_path: BTreeMap<NamePath, usize> = s
        .netlist
        .components
        .iter()
        .enumerate()
        .map(|(i, c)| (c.path.clone(), i))
        .collect();
    let mut by_comp = BTreeMap::new();
    let mut pending = Vec::new();
    for f in board.footprints.values() {
        let hit = f
            .component
            .as_deref()
            .map(|p| follow(s.lock, &NamePath::parse(p)))
            .and_then(|p| by_path.get(&p).copied());
        match hit {
            Some(ci) if !by_comp.contains_key(&ci) => {
                by_comp.insert(ci, f.id);
            }
            _ => pending.push(f.id),
        }
    }
    let mut unlinked = Vec::new();
    for id in pending {
        let f = &board.footprints[&id];
        let by_ref = (o.relink_by_designator && f.component.is_none())
            .then(|| s.netlist.component(&f.reference).map(|(i, _)| i))
            .flatten()
            .filter(|ci| !by_comp.contains_key(ci));
        match by_ref {
            Some(ci) => {
                by_comp.insert(ci, id);
            }
            None => unlinked.push(id),
        }
    }
    Links { by_comp, unlinked }
}

pub(super) fn sheet_of(path: &NamePath, root: &str) -> (String, String) {
    let parent = path.parent().unwrap_or_default();
    if parent.depth() == 0 {
        (root.to_owned(), "/".to_owned())
    } else {
        (parent.0.join("/"), parent.to_string())
    }
}

pub(super) fn comp_cause(c: &CompInst, root: &str, reason: String) -> Cause {
    let (sheet, sheet_path) = sheet_of(&c.path, root);
    Cause {
        sheet,
        sheet_path,
        source: (!c.source.file.is_empty()).then(|| c.source.to_string()),
        reason,
    }
}

pub(super) fn root_cause(root: &str, source: Option<String>, reason: String) -> Cause {
    Cause {
        sheet: root.to_owned(),
        sheet_path: "/".to_owned(),
        source,
        reason,
    }
}

/// A net as the review shows it: a sheet-local net (`/ch[1]/TIE_B`) by its name and sheet
/// (`TIE_B (ch[1])`), never as a raw path.
pub fn net_label(name: &str) -> String {
    match name
        .strip_prefix('/')
        .and_then(|rest| rest.rsplit_once('/'))
    {
        Some((sheet, leaf)) => format!("{leaf} ({sheet})"),
        None => name.to_owned(),
    }
}

pub(super) fn package_name(pool: &Pool, u: Option<Uuid>) -> Option<String> {
    u.and_then(|u| pool.packages.get(&u))
        .map(|p| p.name.clone())
}

/// The room a channel's components belong to (`ch[2]`): the sheet instance with an index.
pub(super) fn channel_of(c: &CompInst) -> Option<String> {
    c.channel?;
    let parent = c.path.parent()?;
    // The shallowest segment carrying the channel index.
    let at = parent.0.iter().rposition(|seg| seg.ends_with(']'))?;
    Some(parent.0[..=at].join("/"))
}

/* ------------------------------------------------------------------------------------------------
 * Board impact
 * --------------------------------------------------------------------------------------------- */

pub(super) fn copper_on_net(board: &Board, net: NetId) -> Impact {
    Impact {
        tracks: board
            .tracks
            .values()
            .filter(|t| t.net == Some(net))
            .map(|t| t.id.0)
            .collect(),
        vias: board
            .vias
            .values()
            .filter(|v| v.net == Some(net))
            .map(|v| v.id.0)
            .collect(),
        zones: board
            .zones
            .values()
            .filter(|z| z.net == Some(net))
            .map(|z| z.id.0)
            .collect(),
        nets: vec![board.net_name(net).to_owned()],
        ..Impact::default()
    }
}

/// Tracks and vias touching a pad on its net (the copper a pin change disconnects).
pub(super) fn copper_at_pad(board: &Board, fid: ItemId, pad: &str) -> Impact {
    let mut i = Impact::default();
    let f = &board.footprints[&fid];
    i.pads.push(format!("{}.{}", f.reference, pad));
    let n = board.copper_count();
    let Some(p) = f.pads(n).into_iter().find(|p| p.name == pad) else {
        return i;
    };
    let Some(net) = p.net else { return i };
    let near = p.prim.bbox().inflate(mm(0.05));
    for t in board.tracks.values().filter(|t| t.net == Some(net)) {
        if !p.layers.contains(&t.layer) {
            continue;
        }
        let touches = t
            .prims()
            .iter()
            .any(|q| q.bbox().intersects(&near) && q.gap(&p.prim) <= 0.0);
        if touches {
            i.tracks.push(t.id.0);
        }
    }
    for v in board.vias.values().filter(|v| v.net == Some(net)) {
        let q = Prim::circle(v.at, v.diameter / 2);
        if q.bbox().intersects(&near) && q.gap(&p.prim) <= 0.0 {
            i.vias.push(v.id.0);
        }
    }
    // The pad and the copper on it, not its whole net (that would light up all of GND).
    i
}

pub(super) fn footprint_impact(board: &Board, fid: ItemId) -> Impact {
    let f = &board.footprints[&fid];
    let mut i = Impact {
        footprints: vec![f.reference.clone()],
        ..Impact::default()
    };
    for pad in f
        .package
        .pads
        .iter()
        .map(|p| p.name.clone())
        .collect::<BTreeSet<_>>()
    {
        i.merge(copper_at_pad(board, fid, &pad));
    }
    i.pads.clear();
    i.tidy();
    i
}

/* ------------------------------------------------------------------------------------------------
 * Placement of new parts
 * --------------------------------------------------------------------------------------------- */

pub(super) fn package_extent(pool: &Pool, u: Option<Uuid>) -> (Coord, Coord) {
    let Some(p) = u.and_then(|u| pool.packages.get(&u)) else {
        return (mm(2.0), mm(2.0));
    };
    let r = Rect::from_points(
        p.courtyard
            .iter()
            .flatten()
            .copied()
            .chain(p.pads.iter().map(|d| d.at)),
    );
    if r.is_empty() {
        (mm(2.0), mm(2.0))
    } else {
        (r.width().max(mm(1.0)), r.height().max(mm(1.0)))
    }
}

/// Lays new parts out: inside their channel's room when the board has it, else in a staging
/// area to the right of the board, one block per room or sheet.
pub(super) struct Placer {
    pub(super) staging: Point,
    pub(super) next_y: Coord,
}

impl Placer {
    pub(super) fn new(board: &Board) -> Self {
        let b = board.outline.bbox();
        let (x, y) = if b.is_empty() {
            (0, 0)
        } else {
            (b.max.x + mm(10.0), b.min.y)
        };
        Placer {
            staging: Point::new(x, y),
            next_y: y,
        }
    }
    /// Grid positions for `sizes` inside `area` (row-major, 1 mm gaps), avoiding `occupied`.
    pub(super) fn grid(area: Rect, sizes: &[(Coord, Coord)], occupied: &[Rect]) -> Vec<Point> {
        let gap = mm(1.0);
        let cell_w = sizes.iter().map(|s| s.0).max().unwrap_or(mm(2.0)) + gap;
        let cell_h = sizes.iter().map(|s| s.1).max().unwrap_or(mm(2.0)) + gap;
        let cols = ((area.width() - gap) / cell_w).max(1);
        let mut out = Vec::new();
        let mut k: Coord = 0;
        while out.len() < sizes.len() && k < 100_000 {
            let (cx, cy) = (k % cols, k / cols);
            k += 1;
            let at = Point::new(
                area.min.x + gap + cx * cell_w + cell_w / 2,
                area.min.y + gap + cy * cell_h + cell_h / 2,
            );
            let cell = Rect::new(
                Point::new(at.x - cell_w / 2, at.y - cell_h / 2),
                Point::new(at.x + cell_w / 2, at.y + cell_h / 2),
            );
            if occupied.iter().any(|r| r.intersects(&cell)) {
                continue;
            }
            out.push(at);
        }
        out
    }
    /// A staging block for `sizes`; returns the positions and the block's outline.
    pub(super) fn stage(&mut self, sizes: &[(Coord, Coord)]) -> (Vec<Point>, Rect) {
        let gap = mm(1.0);
        let cell_w = sizes.iter().map(|s| s.0).max().unwrap_or(mm(2.0)) + gap;
        let cell_h = sizes.iter().map(|s| s.1).max().unwrap_or(mm(2.0)) + gap;
        let cols = (sizes.len() as f64).sqrt().ceil().max(1.0) as Coord;
        let rows = (sizes.len() as Coord + cols - 1) / cols;
        let area = Rect::new(
            Point::new(self.staging.x, self.next_y),
            Point::new(
                self.staging.x + cols * cell_w + gap,
                self.next_y + rows.max(1) * cell_h + gap,
            ),
        );
        self.next_y = area.max.y + mm(4.0);
        (Self::grid(area, sizes, &[]), area)
    }
}

/* ------------------------------------------------------------------------------------------------
 * Forward compare
 * --------------------------------------------------------------------------------------------- */

pub(super) struct Builder {
    pub(super) changes: Vec<Change>,
    pub(super) direction: Direction,
}

impl Builder {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn push(
        &mut self,
        kind: ChangeKind,
        key: &str,
        item: impl Into<String>,
        from: Option<String>,
        to: Option<String>,
        cause: Cause,
        impact: Impact,
        op: Op,
    ) -> &mut Change {
        let mut impact = impact;
        impact.tidy();
        self.changes.push(Change {
            id: format!("{}:{key}", kind.as_str()),
            kind,
            direction: self.direction,
            origin: match self.direction {
                Direction::Forward => Origin::Schematic,
                Direction::Back => Origin::Board,
            },
            item: item.into(),
            from,
            to,
            cause,
            impact,
            requires: Vec::new(),
            conflict: None,
            issues: Vec::new(),
            op,
        });
        self.changes.last_mut().unwrap()
    }
}

/// Sort key that orders `R2` before `R10`.
pub(super) fn natural(s: &str) -> (String, u64, String) {
    let digits_at = s.find(|c: char| c.is_ascii_digit()).unwrap_or(s.len());
    let (head, rest) = s.split_at(digits_at);
    let end = rest
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(rest.len());
    (
        head.to_owned(),
        rest[..end].parse().unwrap_or(0),
        rest[end..].to_owned(),
    )
}

pub(super) fn finish(
    direction: Direction,
    mut changes: Vec<Change>,
    o: &ComparatorOptions,
) -> EcoPlan {
    // An ignored difference type produces no lines, whichever part of the compare found it.
    changes.retain(|c| !o.hides(c.kind));
    changes.sort_by(|a, b| {
        (a.kind.group(), a.kind, natural(&a.item), &a.id).cmp(&(
            b.kind.group(),
            b.kind,
            natural(&b.item),
            &b.id,
        ))
    });
    let ids: BTreeSet<String> = changes.iter().map(|c| c.id.clone()).collect();
    for c in &mut changes {
        c.requires.retain(|r| ids.contains(r) && *r != c.id);
        c.requires.sort();
        c.requires.dedup();
    }
    EcoPlan {
        direction,
        changes,
        options: o.clone(),
    }
}

/// The schematic's net names per component pad (`pad → net`).
pub(super) fn sch_pads(nl: &Netlist, c: &CompInst) -> BTreeMap<String, Option<String>> {
    let mut out = BTreeMap::new();
    for p in &c.pins {
        for pad in &p.pads {
            out.insert(pad.clone(), p.net.map(|n| nl.nets[n].name.clone()));
        }
    }
    out
}

/// [`compare`] with the comparator settings saved on the board (its `sync { }` section).
pub fn compare_saved(s: &Schematic, board: &Board) -> EcoPlan {
    compare(s, board, &board.sync.options)
}

/// [`back_annotate`] with the comparator settings saved on the board.
pub fn back_annotate_saved(s: &Schematic, board: &Board) -> EcoPlan {
    back_annotate(s, board, &board.sync.options)
}
