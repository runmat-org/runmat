//! Sketch profiles and paths as kernel segments: closed loops, path chains, segment samples and
//! slot ties, with the rotation helpers they use.

use super::*;

/// Rotate `v` about unit axis `k` by `angle` (Rodrigues).
pub(super) fn rotate(v: [f64; 3], k: [f64; 3], angle: f64) -> [f64; 3] {
    let k = normalize(k);
    let (s, c) = angle.sin_cos();
    add(
        add(scale(v, c), scale(cross(k, v), s)),
        scale(k, dot(k, v) * (1.0 - c)),
    )
}

pub(super) fn rot_apply(r: &[[f64; 3]; 3], p: [f64; 3]) -> [f64; 3] {
    [
        r[0][0] * p[0] + r[0][1] * p[1] + r[0][2] * p[2],
        r[1][0] * p[0] + r[1][1] * p[1] + r[1][2] * p[2],
        r[2][0] * p[0] + r[2][1] * p[1] + r[2][2] * p[2],
    ]
}

/// A sketch spline as a kernel segment: the interpolating cubic through its solved points.
pub(super) fn spline_segment(
    id: &crate::ids::SketchEntityId,
    count: usize,
    closed: bool,
    solved: &SolvedSketch,
) -> Option<ProfileSegment> {
    let points: Vec<[f64; 2]> = SketchGeometry::spline_roles(count, closed)
        .into_iter()
        .map(|r| solved.point(id, r))
        .collect::<Option<_>>()?;
    let kind = runmat_geometry_kernel::interpolate::interpolate(&points, closed)?;
    Some(ProfileSegment {
        curve: id.clone(),
        kind,
        start: points[0],
        end: if closed { points[0] } else { *points.last()? },
    })
}

/// Closed loops from a solved sketch: explicit loops as listed, or every closed chain of
/// non-construction lines and arcs (circles are loops alone). The outer loop (largest area)
/// comes first.
pub(super) fn build_loops(
    sketch: &Sketch,
    solved: &SolvedSketch,
    explicit: &[Vec<crate::ids::SketchEntityId>],
) -> Result<Vec<Vec<ProfileSegment>>, String> {
    let seg_of = |id: &crate::ids::SketchEntityId| -> Option<ProfileSegment> {
        let e = sketch.entities.iter().find(|e| &e.id == id)?;
        match &e.geometry {
            SketchGeometry::Line { .. } => Some(ProfileSegment {
                curve: id.clone(),
                kind: SegmentKind::Line,
                start: solved.point(id, PointRole::Start)?,
                end: solved.point(id, PointRole::End)?,
            }),
            SketchGeometry::Arc { .. } => Some(ProfileSegment {
                curve: id.clone(),
                kind: SegmentKind::Arc {
                    center: solved.point(id, PointRole::Center)?,
                    clockwise: false,
                },
                start: solved.point(id, PointRole::Start)?,
                end: solved.point(id, PointRole::End)?,
            }),
            SketchGeometry::Circle { .. } => {
                let c = solved.point(id, PointRole::Center)?;
                let r = *solved.radii.get(id)?;
                Some(ProfileSegment {
                    curve: id.clone(),
                    kind: SegmentKind::Circle {
                        center: c,
                        radius: r,
                    },
                    start: [c[0] + r, c[1]],
                    end: [c[0] + r, c[1]],
                })
            }
            SketchGeometry::Spline { points, closed } => spline_segment(id, points.len(), *closed, solved),
            SketchGeometry::Point { .. } => None,
        }
    };
    let close = |a: [f64; 2], b: [f64; 2]| (a[0] - b[0]).abs() < 1e-9 && (a[1] - b[1]).abs() < 1e-9;
    let mut loops: Vec<Vec<ProfileSegment>> = Vec::new();
    if !explicit.is_empty() {
        for l in explicit {
            let mut segs = Vec::new();
            for id in l {
                segs.push(seg_of(id).ok_or_else(|| format!("`{}` is not a profile curve", id))?);
            }
            loops.push(segs);
        }
    } else {
        let mut pool: Vec<ProfileSegment> = sketch
            .entities
            .iter()
            .filter(|e| !e.construction)
            .filter_map(|e| seg_of(&e.id))
            .collect();
        while !pool.is_empty() {
            let first = pool.remove(0);
            if first.is_closed_curve() {
                loops.push(vec![first]);
                continue;
            }
            let start = first.start;
            let mut chain = vec![first];
            loop {
                let end = chain.last().expect("non-empty").end;
                if close(end, start) {
                    break;
                }
                let Some(k) = pool.iter().position(|s| {
                    !s.is_closed_curve()
                        && (close(s.start, end) || close(s.end, end))
                }) else {
                    return Err(format!(
                        "the profile is open at ({:.6}, {:.6}) after `{}`",
                        end[0],
                        end[1],
                        chain.last().expect("non-empty").curve
                    ));
                };
                let mut s = pool.remove(k);
                if !close(s.start, end) {
                    reverse_segment(&mut s);
                }
                chain.push(s);
            }
            loops.push(chain);
        }
    }
    if loops.is_empty() {
        return Err("the sketch has no closed profile".into());
    }
    let area = |l: &Vec<ProfileSegment>| -> f64 {
        l.iter()
            .map(|s| match &s.kind {
                SegmentKind::Circle { radius, .. } => std::f64::consts::PI * radius * radius,
                SegmentKind::Ellipse { .. } | SegmentKind::Spline { .. } => s
                    .polyline(64)
                    .windows(2)
                    .map(|w| (w[0][0] * w[1][1] - w[1][0] * w[0][1]) / 2.0)
                    .sum::<f64>(),
                _ => (s.start[0] * s.end[1] - s.end[0] * s.start[1]) / 2.0,
            })
            .sum::<f64>()
            .abs()
    };
    let outer = (0..loops.len())
        .max_by(|a, b| {
            area(&loops[*a])
                .partial_cmp(&area(&loops[*b]))
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(b.cmp(a))
        })
        .expect("non-empty");
    let o = loops.remove(outer);
    loops.insert(0, o);
    Ok(loops)
}

pub(super) fn reverse_segment(s: &mut ProfileSegment) {
    std::mem::swap(&mut s.start, &mut s.end);
    match &mut s.kind {
        SegmentKind::Arc { clockwise, .. } | SegmentKind::Ellipse { clockwise, .. } => {
            *clockwise = !*clockwise;
        }
        SegmentKind::Spline {
            poles,
            weights,
            knots,
            multiplicities,
            ..
        } => {
            poles.reverse();
            weights.reverse();
            let (lo, hi) = (
                knots.first().copied().unwrap_or(0.0),
                knots.last().copied().unwrap_or(1.0),
            );
            for k in knots.iter_mut() {
                *k = lo + hi - *k;
            }
            knots.reverse();
            multiplicities.reverse();
        }
        SegmentKind::Line | SegmentKind::Circle { .. } => {}
    }
}

/// Points along a segment from start to end: the ends for a line, samples for an arc.
pub(super) fn segment_samples(s: &ProfileSegment) -> Vec<[f64; 2]> {
    match s.kind {
        SegmentKind::Ellipse { .. } | SegmentKind::Spline { .. } => s.polyline(24),
        SegmentKind::Line => vec![s.start, s.end],
        SegmentKind::Arc { center, clockwise } => {
            let r = ((s.start[0] - center[0]).powi(2) + (s.start[1] - center[1]).powi(2)).sqrt();
            let a0 = (s.start[1] - center[1]).atan2(s.start[0] - center[0]);
            let a1 = (s.end[1] - center[1]).atan2(s.end[0] - center[0]);
            let tau = std::f64::consts::TAU;
            let mut sweep = a1 - a0;
            if clockwise {
                while sweep >= -1e-12 {
                    sweep -= tau;
                }
            } else {
                while sweep <= 1e-12 {
                    sweep += tau;
                }
            }
            let n = ((sweep.abs() / (std::f64::consts::PI / 12.0)).ceil() as usize).max(2);
            (0..=n)
                .map(|k| {
                    let a = a0 + sweep * k as f64 / n as f64;
                    [center[0] + r * a.cos(), center[1] + r * a.sin()]
                })
                .collect()
        }
        SegmentKind::Circle { center, radius } => (0..=24)
            .map(|k| {
                let a = std::f64::consts::TAU * k as f64 / 24.0;
                [center[0] + radius * a.cos(), center[1] + radius * a.sin()]
            })
            .collect(),
    }
}

/// A sketch's non-construction curves chained end to end into one path, starting from a free
/// end when the path is open.
pub(super) fn build_path(
    sketch: &Sketch,
    solved: &SolvedSketch,
) -> Result<Vec<ProfileSegment>, String> {
    let close = |a: [f64; 2], b: [f64; 2]| (a[0] - b[0]).abs() < 1e-9 && (a[1] - b[1]).abs() < 1e-9;
    let mut pool: Vec<ProfileSegment> = Vec::new();
    for e in sketch.entities.iter().filter(|e| !e.construction) {
        let seg = match &e.geometry {
            SketchGeometry::Line { .. } => ProfileSegment {
                curve: e.id.clone(),
                kind: SegmentKind::Line,
                start: solved
                    .point(&e.id, PointRole::Start)
                    .ok_or("path line not solved")?,
                end: solved
                    .point(&e.id, PointRole::End)
                    .ok_or("path line not solved")?,
            },
            SketchGeometry::Arc { .. } => ProfileSegment {
                curve: e.id.clone(),
                kind: SegmentKind::Arc {
                    center: solved
                        .point(&e.id, PointRole::Center)
                        .ok_or("path arc not solved")?,
                    clockwise: false,
                },
                start: solved
                    .point(&e.id, PointRole::Start)
                    .ok_or("path arc not solved")?,
                end: solved
                    .point(&e.id, PointRole::End)
                    .ok_or("path arc not solved")?,
            },
            SketchGeometry::Spline { points, closed } => {
                spline_segment(&e.id, points.len(), *closed, solved).ok_or("path spline not solved")?
            }
            _ => continue,
        };
        pool.push(seg);
    }
    if pool.is_empty() {
        return Err("the path sketch has no curves".into());
    }
    // Start at a free end: an endpoint no other curve shares.
    let shared = |p: [f64; 2], me: usize, pool: &[ProfileSegment]| {
        pool.iter()
            .enumerate()
            .any(|(j, s)| j != me && (close(s.start, p) || close(s.end, p)))
    };
    let mut first = 0;
    let mut flip = false;
    'find: for (i, s) in pool.iter().enumerate() {
        if !shared(s.start, i, &pool) {
            first = i;
            break 'find;
        }
        if !shared(s.end, i, &pool) {
            first = i;
            flip = true;
            break 'find;
        }
    }
    let mut seg = pool.remove(first);
    if flip {
        reverse_segment(&mut seg);
    }
    let mut chain = vec![seg];
    while !pool.is_empty() {
        let end = chain.last().expect("non-empty").end;
        let Some(k) = pool
            .iter()
            .position(|s| close(s.start, end) || close(s.end, end))
        else {
            return Err(format!(
                "the path is broken at ({:.6}, {:.6}) after `{}`",
                end[0],
                end[1],
                chain.last().expect("non-empty").curve
            ));
        };
        let mut s = pool.remove(k);
        if !close(s.start, end) {
            reverse_segment(&mut s);
        }
        chain.push(s);
    }
    Ok(chain)
}

/// For each body of a slot tie, a reference that selects it alone: the body carrying a face
/// name no other tied piece has.
pub(super) fn slot_tie_candidates(
    body_ref: &Reference,
    tie: &SlotTie,
    bodies: &[BodyNames],
) -> Vec<Reference> {
    let sets: Vec<BTreeSet<&EntityName>> = tie
        .bodies
        .iter()
        .map(|b| {
            bodies
                .get(*b as usize)
                .map(|n| n.faces.iter().flatten().collect())
                .unwrap_or_default()
        })
        .collect();
    let mut out = Vec::new();
    for (i, mine) in sets.iter().enumerate() {
        let unique = mine.iter().find(|n| {
            sets.iter()
                .enumerate()
                .all(|(j, o)| j == i || !o.contains(*n))
        });
        if let Some(n) = unique {
            out.push(Reference::piece(
                body_ref.clone(),
                vec![Reference::face((*n).clone())],
            ));
        }
    }
    out
}
