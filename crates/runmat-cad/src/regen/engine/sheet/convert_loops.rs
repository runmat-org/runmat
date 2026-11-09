//! The fixed face of a conversion as edge loops: its outline and its holes, each edge a line,
//! a circular arc or a whole circle.
//!
//! The topology summary gives an edge only its end points and length, so a curved edge's
//! circle comes from the face across it: a cylinder whose axis is normal to the fixed face
//! (a rounded corner, a round hole). Any other curved edge is refused by its place on the face.

use super::*;
use crate::refs::history::{SurfaceKind, Topology};

/// What an edge of the fixed face is, in the direction its loop runs.
#[derive(Clone, Copy, Debug)]
pub(super) enum Piece {
    Line,
    /// About `center` (on the fixed face's plane), counter-clockwise about the face normal
    /// unless `clockwise`.
    Arc { center: [f64; 3], radius: f64, clockwise: bool },
    /// A closed circular edge: a round hole by itself.
    Circle { center: [f64; 3], radius: f64 },
}

/// One edge of a loop: its index in the topology, where the loop meets it first, and its shape.
#[derive(Clone, Copy, Debug)]
pub(super) struct LoopEdge {
    pub edge: usize,
    pub from: [f64; 3],
    pub piece: Piece,
}

/// The fixed face's loops: the outline counter-clockwise about the face normal `n`, starting
/// at its least vertex, and the holes clockwise.
pub(super) struct FaceLoops {
    pub outer: Vec<LoopEdge>,
    pub holes: Vec<Vec<LoopEdge>>,
}

/// The circle of curved edge `k`: the axis of a cylinder across it that is normal to the face.
fn circle_of(topo: &Topology, k: usize, face: usize, n: [f64; 3]) -> Option<([f64; 3], f64)> {
    let e = &topo.edges[k];
    let other = if e.faces[0] as usize == face { e.faces[1] } else { e.faces[0] } as usize;
    match topo.faces[other].geom.surface {
        SurfaceKind::Cylinder { axis_point, axis_direction, radius } => {
            let a = normalize(axis_direction);
            if dot(a, n).abs() < 1.0 - 1e-9 {
                return None;
            }
            let s = e.geom.start;
            Some((add(axis_point, scale(a, dot(sub(s, axis_point), a))), radius))
        }
        _ => None,
    }
}

/// The signed sweep about `n` from `a` to `b` round `c`, counter-clockwise in `[0, 2π)`.
fn ccw_sweep(c: [f64; 3], a: [f64; 3], b: [f64; 3], n: [f64; 3]) -> f64 {
    let (u, v) = (sub(a, c), sub(b, c));
    let ang = dot(cross(u, v), n).atan2(dot(u, v));
    if ang < 0.0 {
        ang + std::f64::consts::TAU
    } else {
        ang
    }
}

/// Twice the area a loop encloses, signed by its turn about `n` (counter-clockwise positive).
fn twice_area(lp: &[LoopEdge], ends: &dyn Fn(usize) -> [f64; 3], n: [f64; 3]) -> f64 {
    lp.iter()
        .enumerate()
        .map(|(k, e)| {
            let to = ends(k);
            match e.piece {
                Piece::Line => dot(cross(e.from, to), n),
                Piece::Arc { center, radius, clockwise } => {
                    let s = ccw_sweep(center, e.from, to, n);
                    let sweep = if clockwise { s - std::f64::consts::TAU } else { s };
                    dot(cross(center, sub(to, e.from)), n) + radius * radius * sweep
                }
                Piece::Circle { radius, .. } => std::f64::consts::TAU * radius * radius,
            }
        })
        .sum()
}

/// Reverses a loop: edge `k` of the result is edge `m − 1 − k`, starting where it used to end,
/// an arc turning the other way.
fn reversed(lp: &[LoopEdge]) -> Vec<LoopEdge> {
    let m = lp.len();
    (0..m)
        .map(|k| {
            let e = lp[m - 1 - k];
            LoopEdge {
                edge: e.edge,
                from: lp[(m - k) % m].from,
                piece: match e.piece {
                    Piece::Arc { center, radius, clockwise } => Piece::Arc { center, radius, clockwise: !clockwise },
                    p => p,
                },
            }
        })
        .collect()
}

/// The loops of face `face` from its edges `mine`. Errors name the edge that is neither a
/// line nor a circular arc normal to the face.
pub(super) fn face_loops(
    topo: &Topology,
    face: usize,
    mine: &[usize],
    n: [f64; 3],
    tol: f64,
) -> Result<FaceLoops, String> {
    let close = |a: [f64; 3], b: [f64; 3]| {
        let d = sub(a, b);
        dot(d, d).sqrt() <= 10.0 * tol
    };
    // Each edge's shape in its canonical direction (start to end).
    let mut shapes = Vec::with_capacity(mine.len());
    for &k in mine {
        let g = topo.edges[k].geom;
        let chord = sub(g.end, g.start);
        let c = dot(chord, chord).sqrt();
        // A line's chord is its length (an arc's falls short by about L θ² / 24).
        if (c - g.length).abs() <= 1e-6 * g.length + tol && c > 10.0 * tol {
            shapes.push(Piece::Line);
            continue;
        }
        let Some((center, radius)) = circle_of(topo, k, face, n) else {
            let p = scale(add(g.start, g.end), 0.5e3);
            return Err(format!(
                "the fixed face has an edge near ({:.2}, {:.2}, {:.2}) mm that is neither a line nor a circular arc about the face's normal; only lines and arcs convert",
                p[0], p[1], p[2]
            ));
        };
        if close(g.start, g.end) {
            shapes.push(Piece::Circle { center, radius });
        } else {
            let s = ccw_sweep(center, g.start, g.end, n);
            let clockwise = (radius * s - g.length).abs() > (radius * (std::f64::consts::TAU - s) - g.length).abs();
            shapes.push(Piece::Arc { center, radius, clockwise });
        }
    }
    // Chain the edges into loops.
    let mut used = vec![false; mine.len()];
    let mut loops: Vec<Vec<LoopEdge>> = Vec::new();
    while let Some(first) = (0..mine.len()).find(|&j| !used[j]) {
        used[first] = true;
        let g = topo.edges[mine[first]].geom;
        if let Piece::Circle { .. } = shapes[first] {
            loops.push(vec![LoopEdge { edge: mine[first], from: g.start, piece: shapes[first] }]);
            continue;
        }
        let start = g.start;
        let mut lp = vec![LoopEdge { edge: mine[first], from: start, piece: shapes[first] }];
        let mut at = g.end;
        while !close(at, start) {
            let Some(j) = (0..mine.len()).find(|&j| {
                let g = topo.edges[mine[j]].geom;
                !used[j] && !matches!(shapes[j], Piece::Circle { .. }) && (close(g.start, at) || close(g.end, at))
            }) else {
                return Err("the fixed face's edges do not close into loops".into());
            };
            used[j] = true;
            let g = topo.edges[mine[j]].geom;
            let forward = close(g.start, at);
            let piece = match shapes[j] {
                Piece::Arc { center, radius, clockwise } if !forward => Piece::Arc { center, radius, clockwise: !clockwise },
                p => p,
            };
            lp.push(LoopEdge { edge: mine[j], from: at, piece });
            at = if forward { g.end } else { g.start };
        }
        loops.push(lp);
    }
    let ends = |lp: &Vec<LoopEdge>| {
        let lp = lp.clone();
        move |k: usize| lp[(k + 1) % lp.len()].from
    };
    let area = |lp: &Vec<LoopEdge>| twice_area(lp, &ends(lp), n);
    // The outline encloses the most.
    let outer_at = (0..loops.len())
        .max_by(|&a, &b| area(&loops[a]).abs().partial_cmp(&area(&loops[b]).abs()).unwrap_or(std::cmp::Ordering::Equal))
        .ok_or_else(|| "the fixed face has no edges".to_string())?;
    let mut outer = loops.remove(outer_at);
    if matches!(outer[0].piece, Piece::Circle { .. }) || outer.len() < 2 {
        return Err("the fixed face's outline is a single circle, so it has no straight edge to bend".into());
    }
    if area(&outer) < 0.0 {
        outer = reversed(&outer);
    }
    // Start at the least vertex, as earlier conversions named their outline edges.
    let least = (0..outer.len())
        .min_by(|&a, &b| outer[a].from.partial_cmp(&outer[b].from).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or(0);
    outer.rotate_left(least);
    let holes = loops
        .into_iter()
        .map(|lp| if matches!(lp[0].piece, Piece::Circle { .. }) || area(&lp) < 0.0 { lp } else { reversed(&lp) })
        .collect();
    Ok(FaceLoops { outer, holes })
}
