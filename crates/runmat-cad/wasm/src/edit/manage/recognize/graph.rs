//! Feature recognition on a body's face-adjacency graph: which faces of an imported solid are a
//! prism's caps and sides, holes, pockets, and rounds or chamfers on the prism's corners.
//!
//! The method is the attributed adjacency graph of Joshi and Chang ("Graph-based heuristics for
//! recognition of machined features from a 3D solid model", Computer-Aided Design 20(2), 1988):
//! faces are nodes, shared edges are arcs, and each feature is a pattern of face types and their
//! adjacency. Holes are found by the cylinder-and-bottom pattern of hint-based recognition
//! (Vandenbrande and Requicha, "Spatial reasoning for the automatic recognition of machinable
//! features in solid models", IEEE PAMI 15(12), 1993); constant-radius rounds and chamfers are
//! found as a tangent arc or a symmetric cut between two side faces, the blend pattern of Zhu
//! and Menq ("B-Rep model simplification by automatic fillet/round suppressing for efficient
//! automatic feature recognition", Computer-Aided Design 34(2), 2002).
//!
//! Recognition is exact, never approximate: a face either fits a pattern within [`EPS`] or is
//! listed in [`Left`] with the reason. The caller rebuilds the features and compares them with the
//! import before it writes anything.

use runmat_cad::refs::history::{EdgeCurve, SurfaceKind, Topology};
use std::collections::BTreeSet;
use std::f64::consts::PI;

/// Coincidence tolerance for points and levels (m).
pub const EPS: f64 = 1e-6;
/// Tolerance for unit-vector components.
const ANG: f64 = 1e-7;

/// A profile curve in the sketch plane's coordinates (m).
#[derive(Clone, Debug, PartialEq)]
pub enum Seg {
    Line {
        a: [f64; 2],
        b: [f64; 2],
    },
    /// Counter-clockwise from `a` to `b` about `c`.
    Arc {
        c: [f64; 2],
        a: [f64; 2],
        b: [f64; 2],
    },
    Circle {
        c: [f64; 2],
        r: f64,
    },
}

impl Seg {
    fn ends(&self) -> Option<([f64; 2], [f64; 2])> {
        match self {
            Seg::Line { a, b } | Seg::Arc { a, b, .. } => Some((*a, *b)),
            Seg::Circle { .. } => None,
        }
    }

    /// Points along the curve from `a` to `b` (16 per arc), for areas.
    fn points(&self) -> Vec<[f64; 2]> {
        match self {
            Seg::Line { a, b } => vec![*a, *b],
            Seg::Arc { c, a, b } => {
                let r = dist(*a, *c);
                let (s, mut e) = (angle(*c, *a), angle(*c, *b));
                if e <= s {
                    e += 2.0 * PI;
                }
                (0..=16)
                    .map(|i| {
                        let t = s + (e - s) * i as f64 / 16.0;
                        [c[0] + r * t.cos(), c[1] + r * t.sin()]
                    })
                    .collect()
            }
            Seg::Circle { c, r } => (0..32)
                .map(|i| {
                    let t = 2.0 * PI * i as f64 / 32.0;
                    [c[0] + r * t.cos(), c[1] + r * t.sin()]
                })
                .collect(),
        }
    }
}

/// A corner of the outline that becomes a feature: the two lines it joins (indices into the
/// outline) and the round's radius or the chamfer's distance.
#[derive(Clone, Debug, PartialEq)]
pub struct Corner {
    pub lines: (usize, usize),
    pub chamfer: bool,
    pub size: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct HolePlan {
    pub c: [f64; 2],
    pub r: f64,
    /// Drilled from the top cap (else from the bottom).
    pub from_top: bool,
    /// The cylinder's depth; `None` through all.
    pub depth: Option<f64>,
    /// A drill point's included angle (deg); `None` a flat bottom.
    pub point_angle: Option<f64>,
}

/// A blind cut from a cap down (or up) to a floor: the floor's loops, outer first.
#[derive(Clone, Debug, PartialEq)]
pub struct Pocket {
    pub from_top: bool,
    pub level: f64,
    pub loops: Vec<Vec<Seg>>,
}

/// What the body is: a prism along `axis` (0 x, 1 y, 2 z) from `lo` to `hi`, its outline (with
/// recognized corners already made sharp), through openings, corners, holes and pockets.
#[derive(Clone, Debug, PartialEq)]
pub struct Plan {
    pub axis: usize,
    pub lo: f64,
    pub hi: f64,
    pub outline: Vec<Seg>,
    pub openings: Vec<Vec<Seg>>,
    pub corners: Vec<Corner>,
    pub holes: Vec<HolePlan>,
    pub pockets: Vec<Pocket>,
}

/// A face recognition could not place, and why.
#[derive(Clone, Debug, PartialEq)]
pub struct Left {
    pub face: usize,
    pub why: String,
}

const AXES: [&str; 3] = ["x", "y", "z"];

fn uv(p: [f64; 3], k: usize) -> [f64; 2] {
    [p[(k + 1) % 3], p[(k + 2) % 3]]
}

fn dist(a: [f64; 2], b: [f64; 2]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
}

fn near(a: [f64; 2], b: [f64; 2]) -> bool {
    dist(a, b) < EPS
}

fn angle(c: [f64; 2], p: [f64; 2]) -> f64 {
    (p[1] - c[1]).atan2(p[0] - c[0])
}

/// The counter-clockwise sweep from angle `s` to angle `e`, in [0, 2π).
fn sweep(s: f64, e: f64) -> f64 {
    (e - s).rem_euclid(2.0 * PI)
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Class {
    /// A plane facing along the axis (+1) or against it (-1).
    Cap(f64),
    /// A plane or cylinder parallel to the axis.
    Side,
    /// A full or partial cylinder about a line parallel to the axis.
    Round,
    Oblique,
    Cross,
    Other,
}

fn classify(t: &Topology, f: usize, k: usize) -> Class {
    match t.faces[f].geom.surface {
        SurfaceKind::Plane { normal } if normal[k].abs() > 1.0 - ANG => {
            Class::Cap(normal[k].signum())
        }
        SurfaceKind::Plane { normal } if normal[k].abs() < ANG => Class::Side,
        SurfaceKind::Plane { .. } => Class::Oblique,
        SurfaceKind::Cylinder { axis_direction, .. } if axis_direction[k].abs() > 1.0 - ANG => {
            Class::Round
        }
        SurfaceKind::Cylinder { .. } => Class::Cross,
        SurfaceKind::Other => Class::Other,
    }
}

fn edges_of(t: &Topology, f: usize) -> impl Iterator<Item = usize> + '_ {
    t.edges
        .iter()
        .enumerate()
        .filter(move |(_, e)| e.faces.contains(&(f as u32)))
        .map(|(i, _)| i)
}

fn neighbours(t: &Topology, f: usize) -> BTreeSet<usize> {
    t.face_neighbours(f as u32)
        .into_iter()
        .map(|g| g as usize)
        .collect()
}

/// The edges between the face set `faces` and the rest, as curves in the plane of axis `k`.
fn boundary(t: &Topology, faces: &BTreeSet<usize>, k: usize) -> Result<Vec<Seg>, String> {
    let mut out = Vec::new();
    for e in &t.edges {
        let (a, b) = (e.faces[0] as usize, e.faces[1] as usize);
        if faces.contains(&a) == faces.contains(&b) {
            continue;
        }
        let other = if faces.contains(&a) { b } else { a };
        let (s, n) = (uv(e.geom.start, k), uv(e.geom.end, k));
        out.push(match e.geom.curve {
            EdgeCurve::Line => Seg::Line { a: s, b: n },
            EdgeCurve::Circle { center, radius, .. } => {
                let c = uv(center, k);
                if near(s, n) {
                    Seg::Circle { c, r: radius }
                } else {
                    // Which way round: the side face's centroid lies on the arc's side of the
                    // centre, at the middle of its sweep.
                    let m = uv(t.faces[other].geom.centroid, k);
                    let (sa, na, ma) = (angle(c, s), angle(c, n), angle(c, m));
                    if dist(m, c) > EPS && sweep(sa, ma) < sweep(sa, na) {
                        Seg::Arc { c, a: s, b: n }
                    } else if dist(m, c) > EPS {
                        Seg::Arc { c, a: n, b: s }
                    } else {
                        // No side to read: the arc is as long as the edge.
                        let theta = e.geom.length / radius;
                        if (sweep(sa, na) - theta).abs() < (sweep(na, sa) - theta).abs() {
                            Seg::Arc { c, a: s, b: n }
                        } else {
                            Seg::Arc { c, a: n, b: s }
                        }
                    }
                }
            }
            EdgeCurve::Other => {
                return Err(format!(
                    "face {other}: its edge on the cap is a freeform curve"
                ))
            }
        });
    }
    Ok(out)
}

/// Chains curves into closed loops, each in traversal order.
fn loops(mut segs: Vec<Seg>) -> Result<Vec<Vec<Seg>>, String> {
    let mut out = Vec::new();
    while let Some(first) = segs.pop() {
        let Some((start, mut end)) = first.ends() else {
            out.push(vec![first]);
            continue;
        };
        let mut chain = vec![first];
        while !near(start, end) {
            let next = segs
                .iter()
                .position(|s| s.ends().is_some_and(|(a, b)| near(a, end) || near(b, end)));
            let Some(i) = next else {
                return Err("the cap's edges don't close into loops".into());
            };
            let s = segs.remove(i);
            let (a, b) = s.ends().expect("open curve");
            end = if near(a, end) { b } else { a };
            chain.push(s);
        }
        out.push(chain);
    }
    Ok(out)
}

/// A loop's points in traversal order.
fn trace(lp: &[Seg]) -> Vec<[f64; 2]> {
    let mut pts: Vec<[f64; 2]> = Vec::new();
    for s in lp {
        let mut p = s.points();
        if let (Some(last), Some(first)) = (pts.last(), p.first()) {
            if !near(*last, *first) {
                p.reverse();
            }
        } else if let (Some(first), Some(next)) = (p.first().copied(), lp.get(1)) {
            // The first curve runs toward the second.
            if let Some((a, b)) = next.ends() {
                if near(first, a) || near(first, b) {
                    p.reverse();
                }
            }
        }
        pts.extend(p);
    }
    pts
}

/// The loop's signed area (counter-clockwise positive).
fn area(lp: &[Seg]) -> f64 {
    let p = trace(lp);
    let n = p.len();
    (0..n)
        .map(|i| {
            let (a, b) = (p[i], p[(i + 1) % n]);
            a[0] * b[1] - b[0] * a[1]
        })
        .sum::<f64>()
        / 2.0
}

/// Where the infinite lines through `l1` and `l2` meet.
fn meet(l1: &Seg, l2: &Seg) -> Option<[f64; 2]> {
    let (Seg::Line { a: p, b: p2 }, Seg::Line { a: q, b: q2 }) = (l1, l2) else {
        return None;
    };
    let r = [p2[0] - p[0], p2[1] - p[1]];
    let s = [q2[0] - q[0], q2[1] - q[1]];
    let den = r[0] * s[1] - r[1] * s[0];
    if den.abs() < 1e-18 {
        return None;
    }
    let t = ((q[0] - p[0]) * s[1] - (q[1] - p[1]) * s[0]) / den;
    Some([p[0] + t * r[0], p[1] + t * r[1]])
}

fn length(s: &Seg) -> f64 {
    s.ends().map_or(0.0, |(a, b)| dist(a, b))
}

/// Moves the end of line `l` that touches `at` to `to`.
fn move_end(l: &mut Seg, at: [f64; 2], to: [f64; 2]) {
    if let Seg::Line { a, b } = l {
        if near(*a, at) {
            *a = to;
        } else if near(*b, at) {
            *b = to;
        }
    }
}

/// The end `s` and `t` share.
fn shared(s: &Seg, t: &Seg) -> Option<[f64; 2]> {
    let (a, b) = s.ends()?;
    let (c, d) = t.ends()?;
    [a, b].into_iter().find(|p| near(*p, c) || near(*p, d))
}

/// Rounds and chamfers on the outline's corners, each made sharp: a tangent arc between two
/// lines is a round; a line shorter than both neighbours that cuts a convex corner at equal
/// setbacks is a chamfer.
fn corners(outline: &mut Vec<Seg>) -> Vec<Corner> {
    let ccw = area(outline) > 0.0;
    let mut out = Vec::new();
    let mut i = 0;
    while outline.len() > 3 && i < outline.len() {
        let n = outline.len();
        let (p, q) = ((i + n - 1) % n, (i + 1) % n);
        let (prev, here, next) = (&outline[p], &outline[i], &outline[q]);
        let lines = matches!(prev, Seg::Line { .. }) && matches!(next, Seg::Line { .. });
        let corner = if lines { meet(prev, next) } else { None };
        let (Some(corner), Some(s1), Some(s2)) = (corner, shared(prev, here), shared(here, next))
        else {
            i += 1;
            continue;
        };
        let found = match here {
            Seg::Arc { c, .. } => {
                let r = dist(*c, s1);
                // Tangent: the radius at each end is square to the line it meets.
                let square = |l: &Seg, at: [f64; 2]| {
                    l.ends().is_some_and(|(a, b)| {
                        let d = [b[0] - a[0], b[1] - a[1]];
                        let v = [at[0] - c[0], at[1] - c[1]];
                        (d[0] * v[0] + d[1] * v[1]).abs() / (length(l) * r) < 1e-6
                    })
                };
                (square(prev, s1) && square(next, s2)).then_some((false, r))
            }
            Seg::Line { .. } => {
                let (d1, d2) = (dist(corner, s1), dist(corner, s2));
                // Convex: the corner turns the way the loop runs.
                let turn = {
                    let (a, b) = (
                        [s1[0] - corner[0], s1[1] - corner[1]],
                        [s2[0] - corner[0], s2[1] - corner[1]],
                    );
                    a[0] * b[1] - a[1] * b[0]
                };
                let convex = (turn < 0.0) == ccw;
                let short = length(here) < length(prev) && length(here) < length(next);
                ((d1 - d2).abs() < EPS && short && convex && d1 > EPS).then_some((true, d1))
            }
            Seg::Circle { .. } => None,
        };
        let Some((chamfer, size)) = found else {
            i += 1;
            continue;
        };
        move_end(&mut outline[p], s1, corner);
        move_end(&mut outline[q], s2, corner);
        outline.remove(i);
        // Indices after `i` moved down by one; earlier corners keep theirs.
        for c in out.iter_mut() {
            let Corner { lines, .. } = c;
            if lines.0 > i {
                lines.0 -= 1;
            }
            if lines.1 > i {
                lines.1 -= 1;
            }
        }
        let n = outline.len();
        let a = if p > i { p - 1 } else { p };
        let b = if q > i { q - 1 } else { q };
        out.push(Corner {
            lines: (a % n, b % n),
            chamfer,
            size,
        });
    }
    out
}

/// Recognizes `t` as a prism with holes, pockets and corner features, or lists the faces it
/// can't place.
pub fn recognize(t: &Topology) -> Result<Plan, Vec<Left>> {
    let n = t.faces.len();
    // The axis: the one every face is along or across, with the most cap area.
    let mut best: Option<(usize, f64, Vec<Left>)> = None;
    for (k, axis) in AXES.iter().enumerate() {
        let mut cap = 0.0;
        let mut left = Vec::new();
        for f in 0..n {
            match classify(t, f, k) {
                Class::Cap(_) => cap += t.faces[f].geom.area,
                Class::Oblique => left.push(Left {
                    face: f,
                    why: format!(
                        "a plane at an angle to the {} axis (a draft, or a chamfer along a cap edge)",
                        axis
                    ),
                }),
                Class::Cross => left.push(Left {
                    face: f,
                    why: format!(
                        "a cylinder across the {} axis (a round along a cap edge, or a cross hole)",
                        axis
                    ),
                }),
                _ => {}
            }
        }
        let better = match &best {
            None => true,
            Some((_, c, l)) => left.len() < l.len() || (left.len() == l.len() && cap > *c),
        };
        if better {
            best = Some((k, cap, left));
        }
    }
    let (k, _, mut left) = best.expect("three axes");
    // The caps' levels, from their centroids (exact; bounding boxes carry the kernel's slack).
    let extreme = |sign: f64| {
        (0..n)
            .filter(|f| classify(t, *f, k) == Class::Cap(sign))
            .map(|f| sign * t.faces[f].geom.centroid[k])
            .fold(f64::NEG_INFINITY, f64::max)
            * sign
    };
    let (lo, hi) = (extreme(-1.0), extreme(1.0));
    let reach = t
        .faces
        .iter()
        .any(|f| f.geom.bbox_min[k] < lo - 1e-5 || f.geom.bbox_max[k] > hi + 1e-5);
    if !lo.is_finite() || !hi.is_finite() || reach {
        left.push(Left {
            face: 0,
            why: format!("the body has no flat cap at each end along {}", AXES[k]),
        });
        return Err(left);
    }
    let level = |f: usize, at: f64| (t.faces[f].geom.centroid[k] - at).abs() < EPS;
    let caps = |sign: f64, at: f64| -> BTreeSet<usize> {
        (0..n)
            .filter(|f| classify(t, *f, k) == Class::Cap(sign) && level(*f, at))
            .collect()
    };
    let (bottom, top) = (caps(-1.0, lo), caps(1.0, hi));
    if bottom.is_empty() || top.is_empty() {
        left.push(Left {
            face: 0,
            why: format!("the body has no flat cap at each end along {}", AXES[k]),
        });
        return Err(left);
    }
    let mut claimed: BTreeSet<usize> = bottom.union(&top).copied().collect();

    // Holes: full cylinders along the axis, open at a cap, with a flat or drill-point bottom.
    let mut groups: Vec<([f64; 2], f64, Vec<usize>)> = Vec::new();
    for f in 0..n {
        if let SurfaceKind::Cylinder {
            axis_point, radius, ..
        } = t.faces[f].geom.surface
        {
            if classify(t, f, k) != Class::Round {
                continue;
            }
            let c = uv(axis_point, k);
            match groups
                .iter_mut()
                .find(|(gc, gr, _)| near(*gc, c) && (gr - radius).abs() < EPS)
            {
                Some(g) => g.2.push(f),
                None => groups.push((c, radius, vec![f])),
            }
        }
    }
    let bottom_segs = boundary(t, &bottom, k).map_err(|why| {
        let mut l = left.clone();
        l.push(Left { face: 0, why });
        l
    })?;
    let bottom_loops = loops(bottom_segs).map_err(|why| {
        let mut l = left.clone();
        l.push(Left {
            face: *bottom.iter().next().expect("a bottom cap"),
            why,
        });
        l
    })?;
    let Some(outer_at) = (0..bottom_loops.len()).max_by(|a, b| {
        area(&bottom_loops[*a])
            .abs()
            .total_cmp(&area(&bottom_loops[*b]).abs())
    }) else {
        left.push(Left {
            face: 0,
            why: "the bottom cap has no outline".into(),
        });
        return Err(left);
    };
    // A prism whose outline is one circle: its cylinder is the outline, not a hole.
    let outer_circle = match bottom_loops[outer_at].as_slice() {
        [Seg::Circle { c, r }] => Some((*c, *r)),
        _ => None,
    };
    let mut holes = Vec::new();
    for (c, r, faces) in &groups {
        let at_top = faces
            .iter()
            .flat_map(|f| edges_of(t, *f))
            .filter_map(|e| match t.edges[e].geom.curve {
                EdgeCurve::Circle { center, .. } => Some((center[k], t.edges[e].geom.length)),
                _ => None,
            })
            .collect::<Vec<_>>();
        // Its ends: the levels of its circular edges.
        let z1 = at_top
            .iter()
            .map(|(h, _)| *h)
            .fold(f64::NEG_INFINITY, f64::max);
        let z0 = at_top.iter().map(|(h, _)| *h).fold(f64::INFINITY, f64::min);
        if !z0.is_finite() {
            continue;
        }
        let around: f64 = at_top
            .iter()
            .filter(|(h, _)| (h - z1).abs() < EPS)
            .map(|(_, l)| l)
            .sum();
        if (around - 2.0 * PI * r).abs() > 1e-6 * r.max(1e-3) {
            continue; // part of a cylinder: a side of the outline or a pocket
        }
        if outer_circle.is_some_and(|(oc, or)| near(oc, *c) && (or - r).abs() < EPS) {
            claimed.extend(faces.iter().copied());
            continue; // the prism itself is a cylinder
        }
        let (through, from_top) = (
            (z0 - lo).abs() < EPS && (z1 - hi).abs() < EPS,
            (z1 - hi).abs() < EPS,
        );
        let open_bottom = (z0 - lo).abs() < EPS;
        if through {
            claimed.extend(faces.iter().copied());
            holes.push(HolePlan {
                c: *c,
                r: *r,
                from_top: true,
                depth: None,
                point_angle: None,
            });
            continue;
        }
        if !from_top && !open_bottom {
            for f in faces {
                left.push(Left { face: *f, why: "a cylinder that opens at neither cap (a counterbore, or a hole from a pocket floor)".into() });
            }
            continue;
        }
        // The bottom: a disk facing the opening at the inner end, or a cone past it.
        let inner = if from_top { z0 } else { z1 };
        let sign = if from_top { 1.0 } else { -1.0 };
        let within = |g: usize| {
            let b = &t.faces[g].geom;
            let (lo2, hi2) = (uv(b.bbox_min, k), uv(b.bbox_max, k));
            lo2[0] > c[0] - r - EPS
                && lo2[1] > c[1] - r - EPS
                && hi2[0] < c[0] + r + EPS
                && hi2[1] < c[1] + r + EPS
        };
        let mut bottom_face = None;
        for g in faces.iter().flat_map(|f| neighbours(t, *f)) {
            if faces.contains(&g) || !within(g) {
                continue;
            }
            let b = &t.faces[g].geom;
            match classify(t, g, k) {
                Class::Cap(s) if s == sign && (b.centroid[k] - inner).abs() < EPS => {
                    bottom_face = Some((g, None));
                }
                Class::Other => {
                    let tip = if from_top {
                        inner - b.bbox_min[k]
                    } else {
                        b.bbox_max[k] - inner
                    };
                    if tip > EPS {
                        bottom_face = Some((g, Some(2.0 * (r / tip).atan().to_degrees())));
                    }
                }
                _ => {}
            }
        }
        match bottom_face {
            Some((g, point_angle)) => {
                claimed.extend(faces.iter().copied());
                claimed.insert(g);
                holes.push(HolePlan {
                    c: *c,
                    r: *r,
                    from_top,
                    depth: Some(z1 - z0),
                    point_angle,
                });
            }
            None => {
                for f in faces {
                    left.push(Left {
                        face: *f,
                        why: "a blind cylinder whose bottom is neither flat nor a drill point"
                            .into(),
                    });
                }
            }
        }
    }

    // The outline and through openings: the bottom cap's loops, holes taken out.
    let is_hole = |s: &Seg| matches!(s, Seg::Circle { c, r } if holes.iter().any(|h| near(h.c, *c) && (h.r - r).abs() < EPS));
    let mut openings = bottom_loops;
    let mut outline = openings.remove(outer_at);
    openings.retain(|lp| !(lp.len() == 1 && is_hole(&lp[0])));
    // An inner loop is a through opening when its walls run the full height; otherwise a pocket
    // from the bottom makes it.
    openings.retain(|lp| {
        let pts = trace(lp);
        (0..n).any(|f| {
            classify(t, f, k) == Class::Cap(1.0)
                && level(f, hi)
                && pts.iter().all(|p| {
                    let b = &t.faces[f].geom;
                    let (l2, h2) = (uv(b.bbox_min, k), uv(b.bbox_max, k));
                    p[0] > l2[0] - EPS
                        && p[1] > l2[1] - EPS
                        && p[0] < h2[0] + EPS
                        && p[1] < h2[1] + EPS
                })
        })
    });
    let corners = corners(&mut outline);

    // Pockets: floors facing a cap, between the caps.
    let mut pockets = Vec::new();
    for f in 0..n {
        let Class::Cap(sign) = classify(t, f, k) else {
            continue;
        };
        if claimed.contains(&f) {
            continue;
        }
        let z = t.faces[f].geom.centroid[k];
        if z < lo + EPS || z > hi - EPS {
            left.push(Left {
                face: f,
                why: "a cap facing into the body".into(),
            });
            continue;
        }
        let floor: BTreeSet<usize> = [f].into();
        match boundary(t, &floor, k).and_then(loops) {
            Ok(mut ls) => {
                ls.retain(|lp| !(lp.len() == 1 && is_hole(&lp[0])));
                let outer = (0..ls.len())
                    .max_by(|a, b| area(&ls[*a]).abs().total_cmp(&area(&ls[*b]).abs()))
                    .unwrap_or(0);
                if !ls.is_empty() {
                    ls.swap(0, outer);
                }
                claimed.insert(f);
                pockets.push(Pocket {
                    from_top: sign > 0.0,
                    level: z,
                    loops: ls,
                });
            }
            Err(why) => left.push(Left { face: f, why }),
        }
    }

    // What is left: faces no pattern placed. Sides along the axis are the outline's and the
    // pockets' walls.
    for f in 0..n {
        if claimed.contains(&f) || left.iter().any(|l| l.face == f) {
            continue;
        }
        match classify(t, f, k) {
            Class::Side | Class::Round => {}
            Class::Other => left.push(Left {
                face: f,
                why: "a curved face that is not a drill point (a freeform, cone or torus face)"
                    .into(),
            }),
            _ => {}
        }
    }
    if !left.is_empty() {
        left.sort_by_key(|l| l.face);
        left.dedup_by_key(|l| l.face);
        return Err(left);
    }
    Ok(Plan {
        axis: k,
        lo,
        hi,
        outline,
        openings,
        corners,
        holes,
        pockets,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line(a: [f64; 2], b: [f64; 2]) -> Seg {
        Seg::Line { a, b }
    }

    #[test]
    fn a_rounded_and_a_chamfered_corner_become_sharp_corners() {
        // A 60 x 40 rectangle with a 4 round at (60, 0) and a 2 chamfer at (60, 40).
        let mut outline = vec![
            line([0.0, 0.0], [56.0, 0.0]),
            Seg::Arc {
                c: [56.0, 4.0],
                a: [56.0, 0.0],
                b: [60.0, 4.0],
            },
            line([60.0, 4.0], [60.0, 38.0]),
            line([60.0, 38.0], [58.0, 40.0]),
            line([58.0, 40.0], [0.0, 40.0]),
            line([0.0, 40.0], [0.0, 0.0]),
        ];
        let found = corners(&mut outline);
        assert_eq!(outline.len(), 4, "{outline:?}");
        assert_eq!(outline[0], line([0.0, 0.0], [60.0, 0.0]));
        assert_eq!(outline[1], line([60.0, 0.0], [60.0, 40.0]));
        assert_eq!(outline[2], line([60.0, 40.0], [0.0, 40.0]));
        assert_eq!(found.len(), 2);
        assert!(!found[0].chamfer && (found[0].size - 4.0).abs() < 1e-9);
        assert_eq!(found[0].lines, (0, 1));
        assert!(found[1].chamfer && (found[1].size - 2.0).abs() < 1e-9);
        assert_eq!(found[1].lines, (1, 2));
    }

    #[test]
    fn loops_close_and_their_area_is_signed() {
        let segs = vec![
            line([0.0, 0.0], [2.0, 0.0]),
            line([2.0, 2.0], [0.0, 2.0]),
            line([2.0, 0.0], [2.0, 2.0]),
            line([0.0, 2.0], [0.0, 0.0]),
            Seg::Circle {
                c: [1.0, 1.0],
                r: 0.25,
            },
        ];
        let ls = loops(segs).unwrap();
        assert_eq!(ls.len(), 2);
        let areas: Vec<f64> = ls.iter().map(|l| area(l).abs()).collect();
        assert!(areas.iter().any(|a| (a - 4.0).abs() < 1e-9), "{areas:?}");
    }
}
