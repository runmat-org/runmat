//! `board.teardrops` on a layout's text: a teardrop where a track ends on a pad or via of its
//! net, written as a small solid-connected pour named `teardrop`, so DRC, fills, outputs and
//! the 3D view treat it as the copper it is. Remove takes the teardrops out again.
//!
//! The shape follows the usual proportions: at the pad it is 90 % of the pad's smaller size
//! wide, and it narrows to the track's width half the pad's size past the pad's edge (never past
//! the track's other end). A track at least 80 % as wide as the pad gets none. Each run replaces
//! the teardrops in its scope (the whole board, or the picked tracks, pads and vias), and every
//! teardrop a DRC finding names is left out, so the ones written pass DRC.

use super::edit::{failure, EditError, EditOutput, EditRequest};
use super::*;
use crate::board_rules::BItem;
use crate::geom::poly::Polygon;
use crate::geom::{mm, Coord, Point};
use crate::layer_set::CopperIndex;
use crate::layout::{NetId, PadConnect, Zone};

/// The name teardrop pours carry.
pub const TEARDROP: &str = "teardrop";

/// Runs `board.teardrops`.
pub(super) fn run(project: &Project, request: &EditRequest) -> Result<EditOutput, EditError> {
    let layout = project.layout_or_err()?;
    let values = &request.values;
    let remove = values.get("action").and_then(Value::as_str) == Some("remove");
    let all =
        values.get("all").and_then(Value::as_bool).unwrap_or(true) || request.selection.is_empty();
    let mut summary = String::new();
    let (change, ()) = board_edit::edit_board(&layout.path, &layout.text, |board| {
        let scope = Scope::of(board, if all { &[] } else { &request.selection });
        let old: Vec<ItemId> = board
            .zones
            .values()
            .filter(|z| z.name == TEARDROP && scope.covers_zone(board, z))
            .map(|z| z.id)
            .collect();
        for id in &old {
            board.zones.remove(id);
        }
        if remove {
            summary = format!("Removed {} teardrop(s)", old.len());
            return if old.is_empty() {
                Err("there are no teardrops to remove here".into())
            } else {
                Ok(())
            };
        }
        let shapes = shapes(board, &scope);
        let tried = shapes.len();
        let mut new = Vec::new();
        for (net, layer, ring) in shapes {
            let id = board.alloc();
            let mut z = Zone::new(
                id,
                TEARDROP,
                Some(net),
                vec![layer],
                Polygon::from_ring(ring),
            );
            z.priority = 100;
            z.connect = PadConnect::Solid;
            z.min_width = mm(0.05);
            board.zones.insert(id, z);
            new.push(id);
        }
        let kept = keep_clean(board, new);
        summary = format!(
            "Added {kept} teardrop(s){}",
            if kept < tried {
                format!("; {} failed DRC and were left out", tried - kept)
            } else {
                String::new()
            }
        );
        if kept == 0 {
            return Err("no track end here takes a teardrop that passes DRC".into());
        }
        Ok(())
    })
    .map_err(failure)?;
    Ok(EditOutput {
        summary,
        files: vec![file_change(&layout.path, &layout.text, change)],
    })
}

/// What a run covers: everything, or the picked tracks, pads (by footprint) and vias.
struct Scope {
    all: bool,
    tracks: BTreeSet<ItemId>,
    footprints: BTreeSet<String>,
    vias: BTreeSet<ItemId>,
}

impl Scope {
    fn of(board: &Board, picked: &[String]) -> Scope {
        let mut s = Scope {
            all: picked.is_empty(),
            tracks: BTreeSet::new(),
            footprints: BTreeSet::new(),
            vias: BTreeSet::new(),
        };
        for p in picked {
            let (kind, key) = p.split_once(':').unwrap_or(("", p));
            let id = key.parse::<u64>().ok().map(ItemId);
            match kind {
                "track" => s.tracks.extend(id.filter(|i| board.tracks.contains_key(i))),
                "via" => s.vias.extend(id.filter(|i| board.vias.contains_key(i))),
                "component" | "footprint" => {
                    s.footprints.insert(key.to_owned());
                }
                "pad" => {
                    s.footprints
                        .insert(key.split('.').next().unwrap_or(key).to_owned());
                }
                _ => {}
            }
        }
        s
    }

    fn covers(&self, track: ItemId, pad_ref: Option<&str>, via: Option<ItemId>) -> bool {
        self.all
            || self.tracks.contains(&track)
            || pad_ref.is_some_and(|r| self.footprints.contains(r))
            || via.is_some_and(|v| self.vias.contains(&v))
    }

    /// Whether an existing teardrop is in scope: it sits on a covered track end.
    fn covers_zone(&self, board: &Board, z: &Zone) -> bool {
        self.all
            || board
                .tracks
                .values()
                .filter(|t| t.net.is_some() && t.net == z.net)
                .any(|t| {
                    let (a, b) = t.endpoints();
                    [a, b].into_iter().any(|e| {
                        z.outline.contains(e) != Some(false)
                            && anchor(board, t, e).is_some_and(|(_, _, pad, via)| {
                                self.covers(t.id, pad.as_deref(), via)
                            })
                    })
                })
    }
}

/// The pad or via of `t`'s net that its end `end` sits on, on its layer: its centre, its smaller
/// size, and the footprint or via it belongs to.
fn anchor(
    board: &Board,
    t: &crate::layout::Track,
    end: Point,
) -> Option<(Point, Coord, Option<String>, Option<ItemId>)> {
    let net = t.net?;
    if let Some(v) = board
        .vias
        .values()
        .find(|v| v.net == Some(net) && v.at == end && (v.from..=v.to).contains(&t.layer))
    {
        return Some((v.at, v.diameter, None, Some(v.id)));
    }
    let n = board.copper_count();
    board.footprints.values().find_map(|f| {
        f.pads(n)
            .into_iter()
            .find(|p| p.net == Some(net) && p.center == end && p.layers.contains(&t.layer))
            .map(|p| {
                let bb = p.prim.bbox();
                (
                    p.center,
                    bb.width().min(bb.height()),
                    Some(f.reference.clone()),
                    None,
                )
            })
    })
}

/// One teardrop per track end on a pad or via of its net: (net, layer, ring).
fn shapes(board: &Board, scope: &Scope) -> Vec<(NetId, CopperIndex, Vec<Point>)> {
    let mut out = Vec::new();
    for t in board.tracks.values() {
        let Some(net) = t.net else { continue };
        let (a, b) = t.endpoints();
        for (end, other) in [(a, b), (b, a)] {
            let Some((c, size, pad_ref, via)) = anchor(board, t, end) else {
                continue;
            };
            if !scope.covers(t.id, pad_ref.as_deref(), via) || t.width as f64 >= 0.8 * size as f64 {
                continue;
            }
            let (dx, dy) = ((other.x - c.x) as f64, (other.y - c.y) as f64);
            let len = dx.hypot(dy);
            if len == 0.0 {
                continue;
            }
            let (ux, uy) = (dx / len, dy / len);
            let reach = (size as f64).min(len);
            let (base, tip) = (0.45 * size as f64, t.width as f64 / 2.0);
            let at = |along: f64, side: f64| {
                Point::new(
                    (c.x as f64 + ux * along - uy * side).round() as Coord,
                    (c.y as f64 + uy * along + ux * side).round() as Coord,
                )
            };
            out.push((
                net,
                t.layer,
                vec![
                    at(0.0, base),
                    at(reach, tip),
                    at(reach, -tip),
                    at(0.0, -base),
                ],
            ));
        }
    }
    out
}

/// Takes out every new teardrop a DRC finding names, refilling and re-checking until none is
/// named. Returns how many stay.
fn keep_clean(board: &mut Board, mut new: Vec<ItemId>) -> usize {
    loop {
        let mut b = board.clone();
        let (mut rules, _) = BoardRules::for_board(&b);
        rules.rebuild(&BoardGraph::new(&b, &[]));
        let fills = fill_zones(&mut b, &rules, &[]);
        let drc = Drc::run(&b, &rules, &[], &fills);
        let named: BTreeSet<ItemId> = drc
            .findings()
            .iter()
            .flat_map(|f| f.items.iter())
            .filter_map(|i| match i {
                BItem::Zone(id) => Some(*id),
                _ => None,
            })
            .filter(|id| new.contains(id))
            .collect();
        if named.is_empty() {
            return new.len();
        }
        for id in &named {
            board.zones.remove(id);
        }
        new.retain(|id| !named.contains(id));
    }
}
