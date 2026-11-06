//! Coupled routing (PCB-25): a differential pair routed as one, and a net's length tuned with
//! meanders.
//!
//! **Pair.** Each net of the pair has two pads, one at each end. The pads closest together
//! across the nets form an end. The router plans the pair's centreline between the midpoints
//! of the two ends, around obstacles grown for the whole pair (two tracks and their gap), with
//! the pair's own copper left out. The two tracks are the centreline offset by half the pitch
//! (width plus gap) to each side, so they run parallel at the gap through every 45° bend. A
//! short stub joins each pad to its track. The board's DRC then decides: if any finding names
//! a new track, nothing is added and the finding is the error.
//!
//! **Tuning.** A net shorter than its target gets trombone meanders on its longest straight
//! track: bumps square to the track at a pitch of the width plus twice the clearance, as many
//! as the run holds, with the amplitude that adds the missing length. One side is tried, then
//! the other. DRC decides as above.

use crate::board_rules::{BItem, BoardGraph, BoardRules};
use crate::drc::Drc;
use crate::geom::{Coord, Point};
use crate::layout::{Board, ItemId, NetId, TrackGeom};

use crate::router::{obstacles, walkaround, Geometry, WalkMode};
use std::collections::BTreeSet;

/// A routed pair: the tracks added for each net and their lengths (mm).
#[derive(Clone, Debug, PartialEq)]
pub struct PairRoute {
    pub p_tracks: Vec<ItemId>,
    pub n_tracks: Vec<ItemId>,
    pub p_length: f64,
    pub n_length: f64,
}

impl PairRoute {
    /// The length difference between the nets (mm).
    pub fn skew(&self) -> f64 {
        (self.p_length - self.n_length).abs()
    }
}

/// The net that owns a copper item, for leaving the pair's own copper out of its obstacles.
fn item_net(board: &Board, item: &BItem) -> Option<NetId> {
    match item {
        BItem::Pad(fp, i) => board.footprints.get(fp)?.pads(board.copper_count()).get(*i as usize)?.net,
        BItem::Track(id) => board.tracks.get(id)?.net,
        BItem::Via(id) => board.vias.get(id)?.net,
        _ => None,
    }
}

/// The two pads of `net` that copper on `layer` can reach: centre and half their larger side.
fn ends(board: &Board, net: NetId, layer: u8) -> Result<[(Point, Coord); 2], String> {
    let pads: Vec<(Point, Coord)> = board
        .all_pads()
        .into_iter()
        .filter(|p| p.net == Some(net) && p.layers.contains(&layer))
        .map(|p| {
            let bb = p.prim.bbox();
            (p.center, bb.width().max(bb.height()) / 2)
        })
        .collect();
    match pads[..] {
        [a, b] => Ok([a, b]),
        _ => Err(format!(
            "{} has {} pad(s) on the layer; a pair routes point to point, two pads per net",
            board.nets[net.0 as usize].name,
            pads.len()
        )),
    }
}

/// Where the pair leaves the pads `(p, n)` heading `toward`: each pad's track runs straight
/// out of the pad, square to the line between the pads, then both merge at 45° to the pair's
/// pitch. Returns the two launch points and the centreline's start.
fn launch(p: (Point, Coord), n: (Point, Coord), toward: Point, width: Coord, pitch: f64) -> (Point, Point, Point) {
    let (ax, ay) = ((n.0.x - p.0.x) as f64, (n.0.y - p.0.y) as f64);
    let spread = ax.hypot(ay);
    let (mut dx, mut dy) = if spread > 0.0 { (-ay / spread, ax / spread) } else { (1.0, 0.0) };
    let m = mid(p.0, n.0);
    if dx * ((toward.x - m.x) as f64) + dy * ((toward.y - m.y) as f64) < 0.0 {
        (dx, dy) = (-dx, -dy);
    }
    let out = |q: Point, k: f64| Point::new(q.x + (dx * k).round() as Coord, q.y + (dy * k).round() as Coord);
    let clear = (p.1.max(n.1) + width) as f64;
    let (lp, ln) = (out(p.0, clear), out(n.0, clear));
    (lp, ln, out(mid(lp, ln), ((spread - pitch) / 2.0).max(0.0)))
}

fn mid(a: Point, b: Point) -> Point {
    Point::new((a.x + b.x) / 2, (a.y + b.y) / 2)
}

/// `path` moved `by` to its left (negative: right), with mitred joins, so each segment stays
/// parallel to the path's at that distance.
pub fn offset(path: &[Point], by: f64) -> Vec<Point> {
    let unit = |a: Point, b: Point| {
        let (dx, dy) = ((b.x - a.x) as f64, (b.y - a.y) as f64);
        let l = (dx * dx + dy * dy).sqrt().max(1.0);
        (dx / l, dy / l)
    };
    let normal = |(ux, uy): (f64, f64)| (-uy, ux);
    let at = |p: Point, (nx, ny): (f64, f64), k: f64| Point::new(p.x + (nx * k).round() as Coord, p.y + (ny * k).round() as Coord);
    let n = path.len();
    (0..n)
        .map(|i| {
            let before = (i > 0).then(|| normal(unit(path[i - 1], path[i])));
            let after = (i + 1 < n).then(|| normal(unit(path[i], path[i + 1])));
            match (before, after) {
                (Some(a), Some(b)) => {
                    let (sx, sy) = (a.0 + b.0, a.1 + b.1);
                    let l = (sx * sx + sy * sy).sqrt().max(1e-9);
                    let (mx, my) = (sx / l, sy / l);
                    // The mitre length keeps both neighbouring segments `by` away.
                    let cos = (mx * a.0 + my * a.1).max(0.2);
                    at(path[i], (mx, my), by / cos)
                }
                (Some(v), None) | (None, Some(v)) => at(path[i], v, by),
                (None, None) => path[i],
            }
        })
        .collect()
}

fn length(path: &[Point]) -> f64 {
    path.windows(2).map(|w| w[0].dist(w[1])).sum::<f64>() / 1e6
}

/// Adds `paths` as tracks of their nets, then keeps them only if DRC names none of them.
fn add_checked(board: &mut Board, pairs: &[(NetId, NetId)], layer: u8, width: Coord, paths: &[(NetId, Vec<Point>)]) -> Result<Vec<Vec<ItemId>>, String> {
    let ids: Vec<Vec<ItemId>> = paths
        .iter()
        .map(|(net, path)| path.windows(2).filter(|w| w[0] != w[1]).map(|w| board.add_track(Some(*net), layer, width, w[0], w[1])).collect())
        .collect();
    let new: BTreeSet<ItemId> = ids.iter().flatten().copied().collect();
    let mut b = board.clone();
    let (mut rules, _) = BoardRules::for_board(&b);
    rules.rebuild(&BoardGraph::new(&b, pairs));
    let fills = crate::zone::fill_zones(&mut b, &rules, pairs);
    let drc = Drc::run(&b, &rules, pairs, &fills);
    let hit = drc.findings().into_iter().find(|f| f.items.iter().any(|i| matches!(i, BItem::Track(id) if new.contains(id))) && f.code != "diff-pair-uncoupled" && f.code != "skew" && f.code != "length").cloned();
    if let Some(f) = hit {
        for id in &new {
            board.tracks.remove(id);
        }
        return Err(format!("{} (DRC {} at {:.3}, {:.3} mm)", f.message, f.code, f.at.x as f64 / 1e6, f.at.y as f64 / 1e6));
    }
    Ok(ids)
}

/// Routes the pair `p`/`n` on `layer` with tracks `width` wide `gap` apart (edge to edge).
pub fn route_pair(board: &mut Board, p: NetId, n: NetId, layer: u8, width: Coord, gap: Coord) -> Result<PairRoute, String> {
    if width <= 0 || gap <= 0 {
        return Err("the width and the gap must be more than zero".into());
    }
    let [pa, pb] = ends(board, p, layer)?;
    let [na, nb] = ends(board, n, layer)?;
    // Each P pad pairs with the N pad nearer to it.
    let (n0, n1) = if pa.0.dist(na.0) + pb.0.dist(nb.0) <= pa.0.dist(nb.0) + pb.0.dist(na.0) { (na, nb) } else { (nb, na) };
    let pitch = (width + gap) as f64;
    // At a 45° bend the inner track's mitred corner sits half the pitch / cos 22.5° from the
    // centreline, about 8 % nearer the copper the bend goes round: the obstacles grow by that.
    let span = 2 * width + gap + (0.1 * pitch) as Coord + 2_000;
    let (lp0, ln0, c0) = launch(pa, n0, mid(pb.0, n1.0), width, pitch);
    let (lp1, ln1, c1) = launch(pb, n1, mid(pa.0, n0.0), width, pitch);
    let (mut rules, _) = BoardRules::for_board(board);
    rules.rebuild(&BoardGraph::new(board, &[(p, n)]));
    let region = crate::geom::Rect::new(c0, c1).inflate(crate::geom::mm(10.0) + (c0.dist(c1) as Coord) / 2);
    let mut obs = obstacles(board, &rules, Some(p), layer, span, region, Geometry::Octilinear);
    obs.hulls.retain(|(item, _)| item_net(board, item) != Some(n));
    let obs = crate::router::Obstacles::new(Geometry::Octilinear, obs.hulls);
    let centre = walkaround(&obs, c0, c1, WalkMode::Shortest).ok_or("no path for the pair around the board's copper at this width and gap")?;
    // Each side sits 1 µm further out, so rounding to whole nanometres on the diagonals never
    // brings the tracks inside the gap.
    let half = pitch / 2.0 + 1_000.0;
    let (left, right) = (offset(&centre, half), offset(&centre, -half));
    // P takes the side its launch point is on.
    let (pp, nn) = if lp0.dist(left[0]) <= lp0.dist(right[0]) { (left, right) } else { (right, left) };
    let ends_on = |pad: Point, out: Point, mut path: Vec<Point>, out_end: Point, pad_end: Point| {
        path.splice(0..0, [pad, out]);
        path.extend([out_end, pad_end]);
        path
    };
    let p_path = ends_on(pa.0, lp0, pp, lp1, pb.0);
    let n_path = ends_on(n0.0, ln0, nn, ln1, n1.0);
    let (p_length, n_length) = (length(&p_path), length(&n_path));
    let ids = add_checked(board, &[(p, n)], layer, width, &[(p, p_path), (n, n_path)])?;
    Ok(PairRoute { p_tracks: ids[0].clone(), n_tracks: ids[1].clone(), p_length, n_length })
}

/// The routed length of `net`: tracks and via spans (mm).
pub fn net_track_length(board: &Board, net: NetId) -> f64 {
    crate::route::net_length(board, net) / 1e6
}

/// A tuned net: its length before and after (mm) and the meanders added.
#[derive(Clone, Debug, PartialEq)]
pub struct Tuning {
    pub before: f64,
    pub after: f64,
    pub bumps: usize,
    pub amplitude: f64,
}

/// Trombone points on `a`→`b`: `bumps` bumps of `amp` to the left (`side` 1) or right (-1),
/// each `pitch` wide with `pitch` between, centred on the run.
fn trombone(a: Point, b: Point, bumps: usize, pitch: f64, amp: f64, side: f64) -> Vec<Point> {
    let l = a.dist(b);
    let (ux, uy) = ((b.x - a.x) as f64 / l, (b.y - a.y) as f64 / l);
    let (nx, ny) = (-uy * side, ux * side);
    let used = (2 * bumps - 1) as f64 * pitch;
    let start = (l - used) / 2.0;
    let at = |s: f64, h: f64| Point::new(a.x + (ux * s + nx * h).round() as Coord, a.y + (uy * s + ny * h).round() as Coord);
    let mut out = vec![a];
    for k in 0..bumps {
        let s = start + 2.0 * k as f64 * pitch;
        out.extend([at(s, 0.0), at(s, amp), at(s + pitch, amp), at(s + pitch, 0.0)]);
    }
    out.push(b);
    out
}

/// Tunes `net` to `target` mm within `tolerance` with trombone meanders on its longest
/// straight track.
pub fn tune_length(board: &mut Board, net: NetId, target: f64, tolerance: f64, pairs: &[(NetId, NetId)]) -> Result<Tuning, String> {
    let before = net_track_length(board, net);
    let name = board.nets[net.0 as usize].name.clone();
    let missing = target - before;
    if missing.abs() <= tolerance {
        return Err(format!("{name} is {before:.3} mm, already within {tolerance} mm of {target} mm"));
    }
    if missing < 0.0 {
        return Err(format!("{name} is {before:.3} mm, longer than {target} mm; meanders only add length, so shorten its route"));
    }
    let run = board
        .tracks
        .values()
        .filter(|t| t.net == Some(net) && matches!(t.geom, TrackGeom::Segment { .. }))
        .max_by_key(|t| {
            let (a, b) = t.endpoints();
            a.dist(b) as i64
        })
        .cloned()
        .ok_or_else(|| format!("{name} has no straight track to meander"))?;
    let (a, b) = run.endpoints();
    let (mut rules, _) = BoardRules::for_board(board);
    rules.rebuild(&BoardGraph::new(board, pairs));
    let clearance = rules.max_clearance().max(board.limits.min_clearance);
    let pitch = (run.width + 2 * clearance) as f64;
    let l = a.dist(b);
    // A bump uses two pitches of the run; a pitch is kept clear at each end.
    let most = ((l - 2.0 * pitch) / (2.0 * pitch)).floor().max(0.0) as usize;
    if most == 0 {
        return Err(format!("{name}'s longest straight track is {:.3} mm, too short to meander at this width and clearance", l / 1e6));
    }
    let mut last = String::new();
    // Fewer, taller bumps first; then more, lower ones.
    for bumps in 1..=most {
        let amp = missing * 1e6 / (2.0 * bumps as f64);
        if amp < pitch / 2.0 && bumps > 1 {
            break;
        }
        if amp > 5.0e6 {
            continue;
        }
        for side in [1.0, -1.0] {
            board.tracks.remove(&run.id);
            let path = trombone(a, b, bumps, pitch, amp, side);
            match add_checked(board, pairs, run.layer, run.width, &[(net, path)]) {
                Ok(_) => {
                    let after = net_track_length(board, net);
                    return Ok(Tuning { before, after, bumps, amplitude: amp / 1e6 });
                }
                Err(e) => {
                    board.tracks.insert(run.id, run.clone());
                    last = e;
                }
            }
        }
    }
    Err(if last.is_empty() {
        format!("{name} needs {missing:.3} mm more; its longest straight track can't hold meanders that tall (5 mm at most)")
    } else {
        format!("no meander on {name}'s longest track passes DRC: {last}")
    })
}
