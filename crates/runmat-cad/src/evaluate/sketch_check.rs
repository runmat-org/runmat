//! Check a solved sketch for the feature it will be used by (Extrude, Revolve, a sweep's path,
//! a loft's profile): open ends, branches, overlapping and crossing curves, the number of closed
//! loops, and for Revolve the axis. Each finding names its curves and a point in sketch
//! coordinates, so the sketch can be repaired where it fails.
//!
//! Construction curves and points are left out: they shape the sketch but are no part of a
//! profile or path. A text is closed loops of its own and is counted, not checked.

use crate::model::sketch::{Sketch, SketchGeometry, SolvedSketch};
use crate::regen::engine::{curve_segment, profile_loops, segment_samples};
use runmat_geometry_kernel::op::{ProfileSegment, SegmentKind};
use serde::Serialize;

/// The feature the sketch is checked for.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SketchUse {
    /// Closed loops that bound regions (Extrude, and Revolve's profile).
    Extrude,
    /// Closed loops on one side of an axis.
    Revolve,
    /// One chain of curves end to end, open or closed.
    SweepPath,
    /// One closed loop.
    LoftProfile,
}

impl SketchUse {
    pub fn parse(v: Option<&str>) -> Self {
        match v {
            Some("revolve") => Self::Revolve,
            Some("sweep_path") => Self::SweepPath,
            Some("loft_profile") => Self::LoftProfile,
            _ => Self::Extrude,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Extrude => "Extrude",
            Self::Revolve => "Revolve",
            Self::SweepPath => "Sweep path",
            Self::LoftProfile => "Loft profile",
        }
    }
}

/// What a finding is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SketchProblem {
    /// A curve's end that meets no other curve.
    OpenEnd,
    /// Three or more curve ends at one point.
    Branch,
    /// Two curves that lie along each other for a length (a line drawn twice, an arc over a
    /// circle).
    Overlap,
    /// Two curves that cross, or one that crosses itself, away from their ends.
    Crossing,
    /// No closed loop, or more than one where one is needed.
    LoopCount,
    /// A path in more than one piece.
    Pieces,
    /// No construction line to revolve about.
    NoAxis,
    /// The profile is on both sides of every construction line.
    CrossesAxis,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct SketchFinding {
    pub problem: SketchProblem,
    /// `error` (the feature fails) or `warning`.
    pub severity: &'static str,
    pub message: String,
    /// The sketch curves it is on (entity ids).
    pub curves: Vec<String>,
    /// Where, in sketch coordinates (metres).
    pub at: Option<[f64; 2]>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct SketchCheck {
    pub for_use: SketchUse,
    /// Profile curves checked (construction left out).
    pub curves: usize,
    /// Closed loops found.
    pub loops: usize,
    /// The construction line Revolve can turn the profile about, when one keeps it on one side.
    pub axis: Option<String>,
    pub findings: Vec<SketchFinding>,
}

impl SketchCheck {
    pub fn errors(&self) -> usize {
        self.findings
            .iter()
            .filter(|f| f.severity == "error")
            .count()
    }
}

fn finding(
    problem: SketchProblem,
    severity: &'static str,
    message: String,
    curves: Vec<String>,
    at: Option<[f64; 2]>,
) -> SketchFinding {
    SketchFinding {
        problem,
        severity,
        message,
        curves,
        at,
    }
}

/// Points closer than this (metres) are one point: the tolerance the profile builder joins
/// curve ends with.
const JOIN: f64 = 1e-9;

fn dist(a: [f64; 2], b: [f64; 2]) -> f64 {
    (a[0] - b[0]).hypot(a[1] - b[1])
}

/// A curve as a polyline in sketch coordinates.
fn polyline(s: &ProfileSegment) -> Vec<[f64; 2]> {
    segment_samples(s)
}

/// Where segments `a0 a1` and `b0 b1` cross, strictly inside both.
fn crossing(a0: [f64; 2], a1: [f64; 2], b0: [f64; 2], b1: [f64; 2]) -> Option<[f64; 2]> {
    let r = [a1[0] - a0[0], a1[1] - a0[1]];
    let s = [b1[0] - b0[0], b1[1] - b0[1]];
    let d = r[0] * s[1] - r[1] * s[0];
    if d.abs() < 1e-18 {
        return None;
    }
    let q = [b0[0] - a0[0], b0[1] - a0[1]];
    let t = (q[0] * s[1] - q[1] * s[0]) / d;
    let u = (q[0] * r[1] - q[1] * r[0]) / d;
    let e = 1e-9;
    (t > e && t < 1.0 - e && u > e && u < 1.0 - e).then(|| [a0[0] + t * r[0], a0[1] + t * r[1]])
}

/// Whether two curves overlap along a length: collinear lines sharing more than a point, or
/// arcs and circles on one circle sharing an arc.
fn overlap(a: &ProfileSegment, b: &ProfileSegment) -> Option<[f64; 2]> {
    match (&a.kind, &b.kind) {
        (SegmentKind::Line, SegmentKind::Line) => {
            let d = [a.end[0] - a.start[0], a.end[1] - a.start[1]];
            let len = d[0].hypot(d[1]);
            if len < JOIN {
                return None;
            }
            let u = [d[0] / len, d[1] / len];
            let off = |p: [f64; 2]| (p[0] - a.start[0]) * u[1] - (p[1] - a.start[1]) * u[0];
            if off(b.start).abs() > JOIN * 10.0 || off(b.end).abs() > JOIN * 10.0 {
                return None;
            }
            let along = |p: [f64; 2]| (p[0] - a.start[0]) * u[0] + (p[1] - a.start[1]) * u[1];
            let (b0, b1) = (
                along(b.start).min(along(b.end)),
                along(b.start).max(along(b.end)),
            );
            let (lo, hi) = (b0.max(0.0), b1.min(len));
            (hi - lo > JOIN * 10.0).then(|| {
                let m = (lo + hi) / 2.0;
                [a.start[0] + u[0] * m, a.start[1] + u[1] * m]
            })
        }
        (
            SegmentKind::Arc { center: ca, .. } | SegmentKind::Circle { center: ca, .. },
            SegmentKind::Arc { center: cb, .. } | SegmentKind::Circle { center: cb, .. },
        ) => {
            let ra = match a.kind {
                SegmentKind::Circle { radius, .. } => radius,
                _ => dist(a.start, *ca),
            };
            let rb = match b.kind {
                SegmentKind::Circle { radius, .. } => radius,
                _ => dist(b.start, *cb),
            };
            if dist(*ca, *cb) > JOIN * 10.0 || (ra - rb).abs() > JOIN * 10.0 {
                return None;
            }
            // On one circle: they overlap where a sample inside one lies on the other.
            let pa = polyline(a);
            let pb = polyline(b);
            let near = |p: [f64; 2], line: &[[f64; 2]]| {
                line.windows(2).any(|w| {
                    let (m, n) = (w[0], w[1]);
                    let l = dist(m, n);
                    l > 0.0 && dist(p, m) + dist(p, n) - l < ra * 1e-3
                })
            };
            pa.iter()
                .skip(1)
                .take(pa.len().saturating_sub(2))
                .find(|p| near(**p, &pb))
                .or_else(|| {
                    pb.iter()
                        .skip(1)
                        .take(pb.len().saturating_sub(2))
                        .find(|p| near(**p, &pa))
                })
                .copied()
        }
        _ => None,
    }
}

/// Checks `sketch` (solved as `solved`) for `use_`.
pub fn check_sketch(sketch: &Sketch, solved: &SolvedSketch, use_: SketchUse) -> SketchCheck {
    let mut findings: Vec<SketchFinding> = Vec::new();
    let curves: Vec<ProfileSegment> = sketch
        .entities
        .iter()
        .filter(|e| !e.construction)
        .filter(|e| {
            !matches!(
                e.geometry,
                SketchGeometry::Text { .. } | SketchGeometry::Point { .. }
            )
        })
        .filter_map(|e| curve_segment(sketch, solved, &e.id))
        .collect();
    let texts = sketch
        .entities
        .iter()
        .filter(|e| !e.construction && matches!(e.geometry, SketchGeometry::Text { .. }))
        .count();
    let mm = |p: [f64; 2]| format!("({:.3}, {:.3}) mm", p[0] * 1000.0, p[1] * 1000.0);

    // Curve ends, joined into points.
    let mut points: Vec<([f64; 2], Vec<usize>)> = Vec::new();
    for (i, c) in curves.iter().enumerate() {
        if c.is_closed_curve() {
            continue;
        }
        for p in [c.start, c.end] {
            match points.iter_mut().find(|(q, _)| dist(*q, p) < JOIN) {
                Some((_, at)) => at.push(i),
                None => points.push((p, vec![i])),
            }
        }
    }
    let name = |i: usize| curves[i].curve.0.clone();
    for (p, at) in &points {
        if at.len() == 1 {
            // The nearest other end, so a near-miss reads as a gap to close.
            let nearest = points
                .iter()
                .filter(|(q, _)| dist(*q, *p) >= JOIN)
                .map(|(q, at)| (dist(*q, *p), at[0]))
                .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            let path_end = use_ == SketchUse::SweepPath;
            if path_end {
                continue;
            }
            let gap = nearest
                .map(|(d, j)| {
                    format!(
                        "; the nearest end, of `{}`, is {:.4} mm away",
                        name(j),
                        d * 1000.0
                    )
                })
                .unwrap_or_default();
            findings.push(finding(
                SketchProblem::OpenEnd,
                "error",
                format!(
                    "`{}` ends at {} and meets no other curve{gap}",
                    name(at[0]),
                    mm(*p)
                ),
                vec![name(at[0])],
                Some(*p),
            ));
        } else if at.len() > 2 {
            let names: Vec<String> = at.iter().map(|i| name(*i)).collect();
            findings.push(finding(
                SketchProblem::Branch,
                "error",
                format!(
                    "{} curves meet at {}: {}",
                    at.len(),
                    mm(*p),
                    names
                        .iter()
                        .map(|n| format!("`{n}`"))
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
                names,
                Some(*p),
            ));
        }
    }

    // Overlaps and crossings between curves, and a curve crossing itself.
    let lines: Vec<Vec<[f64; 2]>> = curves.iter().map(polyline).collect();
    for i in 0..curves.len() {
        for j in i + 1..curves.len() {
            if let Some(at) = overlap(&curves[i], &curves[j]) {
                findings.push(finding(
                    SketchProblem::Overlap,
                    "error",
                    format!("`{}` lies along `{}` near {}", name(i), name(j), mm(at)),
                    vec![name(i), name(j)],
                    Some(at),
                ));
                continue;
            }
            let hit = lines[i].windows(2).find_map(|a| {
                lines[j]
                    .windows(2)
                    .find_map(|b| crossing(a[0], a[1], b[0], b[1]))
            });
            if let Some(at) = hit {
                findings.push(finding(
                    SketchProblem::Crossing,
                    "error",
                    format!("`{}` crosses `{}` at {}", name(i), name(j), mm(at)),
                    vec![name(i), name(j)],
                    Some(at),
                ));
            }
        }
        // A curve crossing itself (a spline's loop): pieces that aren't neighbours.
        let l = &lines[i];
        let own = (0..l.len().saturating_sub(1)).find_map(|a| {
            (a + 2..l.len().saturating_sub(1))
                .find_map(|b| crossing(l[a], l[a + 1], l[b], l[b + 1]))
        });
        if let Some(at) = own {
            findings.push(finding(
                SketchProblem::Crossing,
                "error",
                format!("`{}` crosses itself at {}", name(i), mm(at)),
                vec![name(i)],
                Some(at),
            ));
        }
    }

    let loops = profile_loops(sketch, solved).map(|l| l.len()).unwrap_or(0);
    let clean = findings.iter().all(|f| f.severity != "error");
    let mut axis = None;
    match use_ {
        SketchUse::Extrude | SketchUse::Revolve | SketchUse::LoftProfile => {
            if loops == 0 && curves.is_empty() && texts == 0 {
                findings.push(finding(
                    SketchProblem::LoopCount,
                    "error",
                    "the sketch has no curves to make a profile of".into(),
                    Vec::new(),
                    None,
                ));
            } else if loops == 0 && clean {
                findings.push(finding(
                    SketchProblem::LoopCount,
                    "error",
                    "the sketch has no closed loop".into(),
                    Vec::new(),
                    None,
                ));
            } else if use_ == SketchUse::LoftProfile && loops > 1 {
                findings.push(finding(
                    SketchProblem::LoopCount,
                    "error",
                    format!("a loft profile is one closed loop; the sketch has {loops}"),
                    Vec::new(),
                    None,
                ));
            }
            if use_ == SketchUse::Revolve {
                axis = revolve_axis(sketch, solved, &lines, &mut findings);
            }
        }
        SketchUse::SweepPath => {
            let ends = points.iter().filter(|(_, at)| at.len() == 1).count();
            let pieces = pieces(&curves, &points);
            if curves.is_empty() {
                findings.push(finding(
                    SketchProblem::Pieces,
                    "error",
                    "the sketch has no curves to make a path of".into(),
                    Vec::new(),
                    None,
                ));
            } else if pieces > 1 {
                findings.push(finding(
                    SketchProblem::Pieces,
                    "error",
                    format!("a sweep path is one chain of curves end to end; the sketch has {pieces} separate pieces"),
                    Vec::new(),
                    None,
                ));
            } else if ends != 0 && ends != 2 {
                findings.push(finding(
                    SketchProblem::Pieces,
                    "error",
                    format!("a sweep path has two free ends or none; the sketch has {ends}"),
                    Vec::new(),
                    None,
                ));
            }
        }
    }
    SketchCheck {
        for_use: use_,
        curves: curves.len(),
        loops,
        axis,
        findings,
    }
}

/// How many connected pieces the curves form (a closed curve alone is one).
fn pieces(curves: &[ProfileSegment], points: &[([f64; 2], Vec<usize>)]) -> usize {
    let mut parent: Vec<usize> = (0..curves.len()).collect();
    fn root(p: &mut [usize], i: usize) -> usize {
        let mut i = i;
        while p[i] != i {
            p[i] = p[p[i]];
            i = p[i];
        }
        i
    }
    for (_, at) in points {
        for w in at.windows(2) {
            let (a, b) = (root(&mut parent, w[0]), root(&mut parent, w[1]));
            parent[a] = b;
        }
    }
    (0..curves.len())
        .filter(|i| root(&mut parent, *i) == *i)
        .count()
}

/// Revolve's axis: a construction line with the whole profile on one side of it. Reports a
/// missing axis, or a profile on both sides of every construction line.
fn revolve_axis(
    sketch: &Sketch,
    solved: &SolvedSketch,
    lines: &[Vec<[f64; 2]>],
    findings: &mut Vec<SketchFinding>,
) -> Option<String> {
    let axes: Vec<ProfileSegment> = sketch
        .entities
        .iter()
        .filter(|e| e.construction && matches!(e.geometry, SketchGeometry::Line { .. }))
        .filter_map(|e| curve_segment(sketch, solved, &e.id))
        .collect();
    if axes.is_empty() {
        findings.push(finding(
            SketchProblem::NoAxis,
            "warning",
            "no construction line to revolve about: draw one, or pick an axis in Revolve".into(),
            Vec::new(),
            None,
        ));
        return None;
    }
    let mut crossed = Vec::new();
    for a in &axes {
        let d = [a.end[0] - a.start[0], a.end[1] - a.start[1]];
        let len = d[0].hypot(d[1]);
        if len < JOIN {
            continue;
        }
        let side = |p: [f64; 2]| ((p[0] - a.start[0]) * d[1] - (p[1] - a.start[1]) * d[0]) / len;
        let (mut lo, mut hi) = (0.0f64, 0.0f64);
        for p in lines.iter().flatten() {
            let s = side(*p);
            lo = lo.min(s);
            hi = hi.max(s);
        }
        let tol = JOIN * 10.0;
        if lo >= -tol || hi <= tol {
            return Some(a.curve.0.clone());
        }
        crossed.push(a.curve.0.clone());
    }
    findings.push(finding(
        SketchProblem::CrossesAxis,
        "error",
        format!(
            "the profile lies on both sides of {}: a revolved profile stays on one side of its axis",
            crossed
                .iter()
                .map(|c| format!("`{c}`"))
                .collect::<Vec<_>>()
                .join(" and ")
        ),
        crossed,
        None,
    ));
    None
}
