//! Coupled meanders: a differential pair tuned as one, so it keeps its gap and coupling.
//!
//! The pair's longest coupled run (a track of each net, parallel, on one layer, side by side)
//! gets one meander along its centreline, and each net follows it offset by half the pair's
//! pitch, with mitred corners. A rectangular bump of height `amp` then adds `2·amp` to each net
//! alike: the offset lengthens a bump's top on one net by as much as it shortens it on the
//! other, and the legs keep their height. Skew within the pair stays where it was.

use crate::board_rules::{BoardGraph, BoardRules};
use crate::geom::{Coord, Point};
use crate::layout::{Board, NetId, Track, TrackGeom};
use crate::pair::{add_checked, meander, net_track_length, offset, Pattern};

/// A pair's tuning: each net's length before and after (mm), and the meander.
#[derive(Clone, Debug, PartialEq)]
pub struct PairTuning {
    pub before: (f64, f64),
    pub after: (f64, f64),
    pub bumps: usize,
    pub amplitude: f64,
}

/// The longest stretch where a track of `p` and one of `n` run side by side.
pub struct CoupledRun {
    pub p: Track,
    pub n: Track,
    /// The run's centreline, along `p`'s direction.
    pub c0: Point,
    pub c1: Point,
    /// Centre to centre of the two tracks.
    pub pitch: f64,
    /// `p`'s side of the centreline: 1 left, -1 right.
    pub p_side: f64,
}

fn segment(t: &Track) -> Option<(Point, Point)> {
    match t.geom {
        TrackGeom::Segment { a, b } => Some((a, b)),
        TrackGeom::Arc { .. } => None,
    }
}

/// The pair's longest coupled run: parallel tracks on one layer, closer than three of their
/// widths plus the gap rule would put them.
pub fn coupled_run(board: &Board, p: NetId, n: NetId) -> Option<CoupledRun> {
    let mut best: Option<(f64, CoupledRun)> = None;
    for tp in board.tracks.values().filter(|t| t.net == Some(p)) {
        let Some((a, b)) = segment(tp) else { continue };
        let l = a.dist(b);
        if l == 0.0 {
            continue;
        }
        let (ux, uy) = ((b.x - a.x) as f64 / l, (b.y - a.y) as f64 / l);
        for tn in board.tracks.values().filter(|t| t.net == Some(n) && t.layer == tp.layer) {
            let Some((c, d)) = segment(tn) else { continue };
            let ln = c.dist(d);
            if ln == 0.0 || ((ux * (d.y - c.y) as f64 - uy * (d.x - c.x) as f64) / ln).abs() > 0.02 {
                continue;
            }
            // Signed distance of `n`'s line to the left of `p`'s.
            let off = -uy * (c.x - a.x) as f64 + ux * (c.y - a.y) as f64;
            if off.abs() > 3.0 * (tp.width + tn.width) as f64 {
                continue;
            }
            let s = |q: Point| (q.x - a.x) as f64 * ux + (q.y - a.y) as f64 * uy;
            let (lo, hi) = (s(c).min(s(d)).max(0.0), s(c).max(s(d)).min(l));
            if hi - lo <= 0.0 || best.as_ref().is_some_and(|(len, _)| *len >= hi - lo) {
                continue;
            }
            let at = |t: f64, h: f64| Point::new(a.x + (ux * t - uy * h).round() as Coord, a.y + (uy * t + ux * h).round() as Coord);
            best = Some((
                hi - lo,
                CoupledRun { p: tp.clone(), n: tn.clone(), c0: at(lo, off / 2.0), c1: at(hi, off / 2.0), pitch: off.abs(), p_side: -off.signum() },
            ));
        }
    }
    best.map(|(_, run)| run)
}

/// Tunes the pair `p`/`n` together by `missing` mm on each net, within `tolerance`, with
/// `pattern` meanders on their longest coupled run.
pub fn tune_pair(board: &mut Board, p: NetId, n: NetId, missing: f64, pairs: &[(NetId, NetId)], pattern: Pattern) -> Result<PairTuning, String> {
    let names = format!("{}/{}", board.net_name(p), board.net_name(n));
    let before = (net_track_length(board, p), net_track_length(board, n));
    if missing <= 0.0 {
        return Err(format!("{names} is {:.3}/{:.3} mm, already long enough; meanders only add length", before.0, before.1));
    }
    let run = coupled_run(board, p, n).ok_or_else(|| format!("{names} has no coupled run to meander"))?;
    let (mut rules, _) = BoardRules::for_board(board);
    rules.rebuild(&BoardGraph::new(board, pairs));
    let clearance = rules.max_clearance().max(board.limits.min_clearance) as f64;
    let width = run.p.width.max(run.n.width) as f64;
    // A bump holds both nets: the inner one's legs keep a clearance between them, and so do
    // the outer one's between neighbouring bumps.
    let pitch = run.pitch + width + 2.0 * clearance;
    let l = run.c0.dist(run.c1);
    let most = ((l - 2.0 * pitch) / (2.0 * pitch)).floor().max(0.0) as usize;
    if most == 0 {
        return Err(format!("{names}'s longest coupled run is {:.3} mm, too short to meander at this pitch", l / 1e6));
    }
    let (pa, pb) = segment(&run.p).ok_or("coupled run is an arc")?;
    let (na, nb) = segment(&run.n).ok_or("coupled run is an arc")?;
    // `n`'s ends in `p`'s direction of travel.
    let along = |q: Point| (q.x - run.c0.x) as f64 * (run.c1.x - run.c0.x) as f64 + (q.y - run.c0.y) as f64 * (run.c1.y - run.c0.y) as f64;
    let (na, nb) = if along(na) <= along(nb) { (na, nb) } else { (nb, na) };
    let mut last = String::new();
    for bumps in 1..=most {
        let amp = pattern.amplitude(missing * 1e6, bumps, pitch);
        if amp < pitch / 2.0 + run.pitch && bumps > 1 {
            break;
        }
        if amp > 5.0e6 {
            continue;
        }
        for side in [1.0, -1.0] {
            let centre = meander(run.c0, run.c1, pattern, bumps, pitch, amp, side);
            let path = |start: Point, end: Point, s: f64| {
                let mut v = vec![start];
                v.extend(offset(&centre, s * run.pitch / 2.0));
                v.push(end);
                v.dedup();
                v
            };
            board.tracks.remove(&run.p.id);
            board.tracks.remove(&run.n.id);
            let paths = [(p, path(pa, pb, run.p_side)), (n, path(na, nb, -run.p_side))];
            match add_checked(board, pairs, run.p.layer, run.p.width, &paths) {
                Ok(_) => {
                    let after = (net_track_length(board, p), net_track_length(board, n));
                    return Ok(PairTuning { before, after, bumps, amplitude: amp / 1e6 });
                }
                Err(e) => {
                    board.tracks.insert(run.p.id, run.p.clone());
                    board.tracks.insert(run.n.id, run.n.clone());
                    last = e;
                }
            }
        }
    }
    Err(if last.is_empty() {
        format!("{names} needs {missing:.3} mm more each; its coupled run can't hold meanders that tall (5 mm at most)")
    } else {
        format!("no coupled meander on {names} passes DRC: {last}")
    })
}
