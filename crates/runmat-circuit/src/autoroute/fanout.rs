//! Dog-bone fanout: a short track from an SMD pad to a through via.

use super::space::Space;
use crate::connectivity::layer_mask;
use crate::geom::shape::Prim;
use crate::geom::{mm, Coord, Point};
use crate::layout::{Board, ItemId, NetId, PlacedPad};
use std::collections::{BTreeMap, BTreeSet};

/// What [`fanout`] fans out and with what.
#[derive(Clone, Debug)]
pub struct FanoutOptions {
    /// Footprints to fan out.
    pub footprints: Vec<ItemId>,
    /// Nets whose pads get a via (the plane nets, and for a BGA its escaping signals).
    pub nets: BTreeSet<NetId>,
    /// Via diameter and drill.
    pub via: (Coord, Coord),
    /// Track width per net, and for the rest.
    pub widths: BTreeMap<NetId, Coord>,
    pub default_width: Coord,
    pub clearance: Coord,
}

/// What [`fanout`] did.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct FanoutReport {
    /// Pads given a via.
    pub fanned: u32,
    /// Pads that already had one.
    pub kept: u32,
    /// Pads with no clear site for a via: `reference.pad`.
    pub blocked: Vec<String>,
}

/// Fans out the pads of `opts.footprints` on `opts.nets`.
pub fn fanout(board: &mut Board, opts: &FanoutOptions) -> FanoutReport {
    let mut space = Space::new(board, opts.clearance);
    fanout_in(board, &mut space, opts)
}

/// [`fanout`] with a [`Space`] the caller built.
pub fn fanout_in(board: &mut Board, space: &mut Space, opts: &FanoutOptions) -> FanoutReport {
    let mut report = FanoutReport::default();
    let n = board.copper_count();
    for id in &opts.footprints {
        let Some(fp) = board.footprints.get(id) else {
            continue;
        };
        let bga = fp.package.name.starts_with("BGA");
        let centre = fp.at;
        let reference = fp.reference.clone();
        let pitch = pitch(&fp.pads(n));
        let mut pads: Vec<PlacedPad> = fp
            .pads(n)
            .into_iter()
            .filter(|p| p.layers.len() == 1 && p.net.is_some_and(|x| opts.nets.contains(&x)))
            .collect();
        // Outer pads first: they have the most room, and leave the inner ones the diagonals.
        pads.sort_by_key(|p| std::cmp::Reverse(manhattan(p.center, centre)));
        for pad in pads {
            let net = pad.net.expect("filtered");
            if joined(board, &pad) {
                report.kept += 1;
                continue;
            }
            let width = opts.widths.get(&net).copied().unwrap_or(opts.default_width);
            let layer = *pad.layers.iter().next().expect("one layer");
            let sites = if bga {
                bga_sites(&pad, centre, pitch)
            } else {
                pad_sites(&pad, centre, opts.via.0)
            };
            let (d, drill) = opts.via;
            let site = sites.into_iter().find(|v| {
                space.free(Some(net), 1 << layer, &Prim::segment(pad.center, *v, width), 0)
                    && space.free(Some(net), layer_mask(0..n), &Prim::circle(*v, d / 2), 0)
                    && space.hole_free(*v, drill / 2)
            });
            match site {
                Some(v) => {
                    board.add_track(Some(net), layer, width, pad.center, v);
                    space.add(Some(net), 1 << layer, Prim::segment(pad.center, v, width));
                    board.add_via(Some(net), v, d, drill);
                    space.add(Some(net), layer_mask(0..n), Prim::circle(v, d / 2));
                    space.add_hole(v, drill / 2);
                    report.fanned += 1;
                }
                None => report.blocked.push(format!("{reference}.{}", pad.name)),
            }
        }
    }
    report
}

fn manhattan(a: Point, b: Point) -> Coord {
    (a.x - b.x).abs() + (a.y - b.y).abs()
}

/// The smallest distance between two pad centres (the ball pitch of a BGA).
fn pitch(pads: &[PlacedPad]) -> Coord {
    let mut best = Coord::MAX;
    for (i, a) in pads.iter().enumerate().take(64) {
        for b in &pads[i + 1..] {
            let d = a.center.dist(b.center) as Coord;
            if d > 0 && d < best {
                best = d;
            }
        }
    }
    if best == Coord::MAX {
        mm(1.0)
    } else {
        best
    }
}

/// Whether the pad already has a via of its net inside it or a track of its net ending in it.
fn joined(board: &Board, pad: &PlacedPad) -> bool {
    let net = pad.net;
    board
        .vias
        .values()
        .any(|v| v.net == net && pad.prim.contains(v.at))
        || board.tracks.values().any(|t| {
            let (a, b) = t.endpoints();
            t.net == net
                && pad.layers.contains(&t.layer)
                && (pad.prim.contains(a) || pad.prim.contains(b))
        })
}

/// The four diagonals between balls, the one pointing away from the package centre first.
fn bga_sites(pad: &PlacedPad, centre: Point, pitch: Coord) -> Vec<Point> {
    let rel = pad.center - centre;
    let sx = if rel.x >= 0 { 1 } else { -1 };
    let sy = if rel.y >= 0 { 1 } else { -1 };
    let h = pitch / 2;
    [(sx, sy), (sx, -sy), (-sx, sy), (-sx, -sy)]
        .iter()
        .map(|(fx, fy)| pad.center + Point::new(fx * h, fy * h))
        .collect()
}

/// Sites past the pad's outer end (away from the footprint centre along the pad's long
/// axis), then beside it, then diagonally.
fn pad_sites(pad: &PlacedPad, centre: Point, via_d: Coord) -> Vec<Point> {
    let bb = pad.prim.bbox();
    let (w, h) = (bb.width(), bb.height());
    let rel = pad.center - centre;
    // Outward along the axis the pad points along from the body; for a pad at the centre
    // (an exposed pad) any side.
    let along_x = if rel.x == 0 && rel.y == 0 {
        w >= h
    } else {
        rel.x.abs() >= rel.y.abs()
    };
    let sign = |v: Coord| if v >= 0 { 1 } else { -1 };
    let half = if along_x { w / 2 } else { h / 2 };
    let side = if along_x { h / 2 } else { w / 2 };
    let out = |d: Coord| {
        if along_x {
            Point::new(sign(rel.x) * d, 0)
        } else {
            Point::new(0, sign(rel.y) * d)
        }
    };
    let across = |d: Coord| {
        if along_x {
            Point::new(0, d)
        } else {
            Point::new(d, 0)
        }
    };
    let mut sites = Vec::new();
    let step = mm(0.05);
    let start = half + via_d / 2 + mm(0.1);
    for k in 0..24 {
        sites.push(pad.center + out(start + k * step));
    }
    let beside = side + via_d / 2 + mm(0.1);
    for k in 0..16 {
        sites.push(pad.center + across(beside + k * step));
        sites.push(pad.center - across(beside + k * step));
    }
    for k in 0..12 {
        let d = start + k * step;
        sites.push(pad.center + out(d) + across(d));
        sites.push(pad.center + out(d) - across(d));
    }
    sites
}
