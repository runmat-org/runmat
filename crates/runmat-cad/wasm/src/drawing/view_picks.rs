//! What the sheet picks in a view, and the views drawn on other views' axes:
//!
//! - [`picks`]: each model edge projected into the view (view mm) with the reference a click
//!   on it means: the face it shows edge-on (a planar side, a hole's wall seen end-on) where it
//!   shows one, else the edge itself; on an assembly view, the component (`occurrence:<path>`);
//! - [`auxiliary_axes`]: an auxiliary view looks square to the face its edge shows edge-on in
//!   the parent view;
//! - [`broken_out`]: a broken-out section draws the parent's lines outside its boundary and the
//!   cut at its depth inside it.

use super::values;
use crate::hlr::{Axes, ViewLines};
use serde_json::{json, Value};

type P3 = [f64; 3];

fn dot(a: P3, b: P3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: P3, b: P3) -> P3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

fn unit(a: P3) -> Option<P3> {
    let n = dot(a, a).sqrt();
    (n > 1e-9).then(|| a.map(|x| x / n))
}

/// A number array of a display, inline or a mesh reference read in place
/// ([`runmat_cad::display::wire::numbers`]).
fn floats(v: &Value) -> Vec<f64> {
    runmat_cad::display::wire::numbers(v)
}

fn vec3(v: &Value) -> Option<P3> {
    let f = floats(v);
    (f.len() == 3).then(|| [f[0], f[1], f[2]])
}

/// The face of `faces` a line in the view shows: a plane seen edge-on, else a cylinder seen
/// along its axis (a hole's rim) or across it (its outline).
fn shown<'v>(faces: &[&'v Value], depth: P3) -> Option<&'v str> {
    let plane = |f: &&Value| f["surface"]["surface"] == "plane" && vec3(&f["surface"]["normal"]).is_some_and(|n| dot(n, depth).abs() < 1e-3);
    let cylinder = |f: &&Value| values::cylinder(&f["surface"]).is_some();
    faces.iter().copied().find(plane).or_else(|| faces.iter().copied().find(cylinder)).and_then(|f| f["reference"].as_str())
}

/// One pick per model edge seen in the view: its reference and its points (view mm).
pub fn picks(display: &Value, axes: &Axes) -> Value {
    let project = |p: P3| [dot(p, axes.u), dot(p, axes.v)];
    let mut out = Vec::new();
    let mut part = |bodies: &Value, t: Option<&[f64]>, occurrence: Option<&str>| {
        for body in bodies.as_array().into_iter().flatten() {
            let faces: &[Value] = body["faces"].as_array().map(Vec::as_slice).unwrap_or_default();
            for e in body["edges"].as_array().into_iter().flatten() {
                let pts = floats(&e["points"]);
                let points: Vec<f64> = pts
                    .chunks_exact(3)
                    .flat_map(|c| {
                        let mut p = [c[0], c[1], c[2]];
                        if let Some(t) = t.filter(|t| t.len() >= 12) {
                            p = [0, 1, 2].map(|i| t[3 * i] * p[0] + t[3 * i + 1] * p[1] + t[3 * i + 2] * p[2] + t[9 + i]);
                        }
                        project(p.map(|x| x * 1000.0))
                    })
                    .collect();
                if points.len() < 4 {
                    continue;
                }
                // A segment seen end-on draws as a point; nothing to click.
                let span = points.chunks_exact(2).fold((f64::MAX, f64::MIN, f64::MAX, f64::MIN), |b, p| (b.0.min(p[0]), b.1.max(p[0]), b.2.min(p[1]), b.3.max(p[1])));
                if (span.1 - span.0).max(span.3 - span.2) < 1e-3 {
                    continue;
                }
                let reference = match occurrence {
                    Some(path) => format!("occurrence:{path}"),
                    None => {
                        let on: Vec<&Value> = floats(&e["faces"]).iter().filter_map(|i| faces.get(*i as usize)).collect();
                        match shown(&on, axes.depth).or_else(|| e["reference"].as_str()) {
                            Some(r) => r.to_owned(),
                            None => continue,
                        }
                    }
                };
                out.push(json!({ "reference": reference, "points": points }));
            }
        }
    };
    let data = &display["data"];
    if display["kind"] == "assembly" {
        let prototypes: &[Value] = data["prototypes"].as_array().map(Vec::as_slice).unwrap_or_default();
        for occ in data["occurrences"].as_array().into_iter().flatten() {
            let t = floats(&occ["transform"]);
            let Some(proto) = occ["prototype"].as_u64().and_then(|i| prototypes.get(i as usize)) else { continue };
            let path = occ["path"].as_str().unwrap_or_default();
            if proto["shape"] == "mesh" && !path.is_empty() {
                part(&proto["part"]["bodies"], Some(&t), Some(path));
            }
        }
    } else {
        part(&data["bodies"], None, None);
    }
    Value::Array(out)
}

/// An auxiliary view's axes: looking square to the face `reference` names, which the parent
/// shows edge-on, with the line it shows in the parent running up the view.
pub fn auxiliary_axes(display: &Value, parent: &Axes, reference: &str) -> Option<Axes> {
    let (_, surface) = values::face(display, reference)?;
    let n = unit(vec3(&surface["normal"])?)?;
    if dot(n, parent.depth).abs() > 1e-3 {
        return None;
    }
    let depth = n.map(|x| -x);
    let along = unit(cross(n, parent.depth))?;
    let u = unit(cross(depth, along))?;
    Some(Axes { u, v: cross(u, depth), depth })
}

fn inside(poly: &[[f64; 2]], p: [f64; 2]) -> bool {
    let mut c = false;
    let n = poly.len();
    for i in 0..n {
        let (a, b) = (poly[i], poly[(i + n - 1) % n]);
        if (a[1] > p[1]) != (b[1] > p[1]) && p[0] < (b[0] - a[0]) * (p[1] - a[1]) / (b[1] - a[1]) + a[0] {
            c = !c;
        }
    }
    c
}

/// The runs of `lines` inside (`keep_inside`) or outside `poly`, split every half millimetre.
fn clip(lines: &[Vec<f64>], poly: &[[f64; 2]], keep_inside: bool) -> Vec<Vec<f64>> {
    let mut out = Vec::new();
    for line in lines {
        let mut run: Vec<f64> = Vec::new();
        for w in line.chunks_exact(2).collect::<Vec<_>>().windows(2) {
            let (a, b) = ([w[0][0], w[0][1]], [w[1][0], w[1][1]]);
            let steps = ((((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2)).sqrt() / 0.5).ceil() as usize).clamp(1, 256);
            for s in 0..steps {
                let at = |t: f64| [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];
                let (p, q) = (at(s as f64 / steps as f64), at((s + 1) as f64 / steps as f64));
                let mid = [(p[0] + q[0]) / 2.0, (p[1] + q[1]) / 2.0];
                if inside(poly, mid) == keep_inside {
                    if run.is_empty() {
                        run.extend(p);
                    }
                    run.extend(q);
                } else if run.len() >= 4 {
                    out.push(std::mem::take(&mut run));
                } else {
                    run.clear();
                }
            }
        }
        if run.len() >= 4 {
            out.push(run);
        }
    }
    out
}

/// The part of the region `ring` inside the boundary `poly` (Sutherland–Hodgman: exact for a
/// convex boundary, as a picked broken-out outline nearly always is).
fn within(ring: &[[f64; 2]], poly: &[[f64; 2]]) -> Vec<[f64; 2]> {
    let area: f64 = (0..poly.len()).map(|i| {
        let (a, b) = (poly[i], poly[(i + 1) % poly.len()]);
        a[0] * b[1] - b[0] * a[1]
    }).sum();
    let sign = if area >= 0.0 { 1.0 } else { -1.0 };
    let mut out = ring.to_vec();
    for i in 0..poly.len() {
        let (a, b) = (poly[i], poly[(i + 1) % poly.len()]);
        let side = |p: [f64; 2]| sign * ((b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]));
        let input = std::mem::take(&mut out);
        for k in 0..input.len() {
            let (p, q) = (input[k], input[(k + 1) % input.len()]);
            let (sp, sq) = (side(p), side(q));
            if sp >= 0.0 {
                out.push(p);
            }
            if (sp >= 0.0) != (sq >= 0.0) {
                let t = sp / (sp - sq);
                out.push([p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t]);
            }
        }
        if out.is_empty() {
            break;
        }
    }
    out
}

/// The centre of a view's lines (view mm): the point the sheet puts at the view's `at`.
pub fn centre(lines: &ViewLines) -> [f64; 2] {
    let (mut lo, mut hi) = ([f64::MAX; 2], [f64::MIN; 2]);
    for l in lines.visible.iter().chain(&lines.hidden) {
        for p in l.chunks_exact(2) {
            for k in 0..2 {
                lo[k] = lo[k].min(p[k]);
                hi[k] = hi[k].max(p[k]);
            }
        }
    }
    if lo[0] > hi[0] {
        return [0.0, 0.0];
    }
    [(lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0]
}

/// A broken-out section: `parent`'s lines outside `boundary` (view mm about the view's
/// centre), and inside it the model cut `depth` mm in from its nearest point, hatched.
pub fn broken_out(display: &Value, axes: &Axes, parent: &ViewLines, boundary: &[[f64; 2]], depth: f64) -> ViewLines {
    let c = centre(parent);
    let poly: Vec<[f64; 2]> = boundary.iter().map(|p| [p[0] + c[0], p[1] + c[1]]).collect();
    let b = &display["data"]["bounds"];
    let (lo, hi) = (vec3(&b["min"]).unwrap_or([0.0; 3]), vec3(&b["max"]).unwrap_or([0.0; 3]));
    let nearest = (0..8)
        .map(|k| [0, 1, 2].map(|a| if (k >> a) & 1 == 1 { hi[a] } else { lo[a] } * 1000.0))
        .min_by(|p, q| dot(*p, axes.depth).total_cmp(&dot(*q, axes.depth)))
        .unwrap_or([0.0; 3]);
    let point = [0, 1, 2].map(|i| nearest[i] + axes.depth[i] * depth);
    let cut = crate::hlr::section_lines(display, axes, point);
    let mut out = ViewLines {
        visible: clip(&parent.visible, &poly, false),
        hidden: clip(&parent.hidden, &poly, false),
        centers: parent.centers.clone(),
        hatch: Vec::new(),
    };
    out.visible.extend(clip(&cut.visible, &poly, true));
    // The cut regions inside the boundary, and the boundary itself (a thin freehand line).
    out.hatch = cut
        .hatch
        .iter()
        .filter_map(|h| {
            let ring: Vec<[f64; 2]> = h.chunks_exact(2).map(|p| [p[0], p[1]]).collect();
            let kept = within(&ring, &poly);
            (kept.len() >= 3).then(|| kept.iter().flat_map(|p| [p[0], p[1]]).collect())
        })
        .collect();
    let mut ring: Vec<f64> = poly.iter().flat_map(|p| [p[0], p[1]]).collect();
    ring.extend(poly.first().map(|p| [p[0], p[1]]).unwrap_or_default());
    out.visible.push(ring);
    out
}
