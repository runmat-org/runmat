//! `assembly.hole-alignment` and `assembly.hole-check`: holes in different parts that meet,
//! measured on the assembly's display (each part's faces where the kernel puts them, assembly
//! features applied).
//!
//! A hole is a concave cylindrical face: the kernel's cylinder (axis and radius) when it knows
//! it, otherwise the frame fitted to the face's mesh ([`frames::face_frame`]), concave when the
//! surface's normals point at the axis. Faces of one part on the same axis with the same radius
//! are one hole (a cylinder split at its seam). Two holes in different parts meet when their axes
//! are parallel, their bores overlap seen along the axis, and their lengths along it touch or
//! come within 1 mm (a stack of parts). They line up when their axes are within
//! `values.tolerance` (mm) of each other.
//!
//! Hole check adds each fastener of the assembly's fastener sets: the holes on its axis, the
//! hole diameter against the fastener's, how far its axis sits from the holes' axes, and the
//! stack of parts it passes through against the grip its size was chosen for.
//!
//! `values.components` (occurrence Outline IDs) limits the pairs to those with one of them.
//! Fastener parts themselves (a nut's bore) aren't holes to align.

use super::{Analysis, ModelDimension, Row};
use crate::edit::assembly::frames::{self, cross, dot, norm, scale, sub, unit, FrameKind, V3};
use crate::runtime::Files;
use serde_json::{json, Map, Value};
use std::collections::BTreeSet;

/// How far apart along the axis two parts' holes may be and still meet (m).
const STACK_GAP: f64 = 0.001;
/// Axes this close to parallel (cosine) are one direction.
const PARALLEL: f64 = 0.9998;

#[derive(Clone, Debug)]
struct Hole {
    /// The occurrence's path and Outline ID.
    path: String,
    item: String,
    /// The face picks that make the hole.
    faces: Vec<String>,
    /// The axis, in assembly coordinates (m): a point on it and its unit direction.
    point: V3,
    axis: V3,
    radius: f64,
    /// The hole's extent along `axis` from `point`.
    from: f64,
    to: f64,
}

impl Hole {
    fn diameter_mm(&self) -> f64 {
        self.radius * 2000.0
    }
    /// The distance from `p` to this hole's axis.
    fn off_axis(&self, p: V3) -> f64 {
        let d = sub(p, self.point);
        norm(sub(d, scale(self.axis, dot(d, self.axis))))
    }
    /// `p`'s position along the axis.
    fn along(&self, p: V3) -> f64 {
        dot(sub(p, self.point), self.axis)
    }
    /// The point on the axis at `t`.
    fn at(&self, t: f64) -> V3 {
        frames::add(self.point, scale(self.axis, t))
    }
}

fn points(values: &Value) -> Vec<V3> {
    let flat: Vec<f64> = values
        .as_array()
        .map(|a| a.iter().filter_map(Value::as_f64).collect())
        .unwrap_or_default();
    flat.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect()
}

fn v3(v: &Value) -> Option<V3> {
    let a = v.as_array()?;
    Some([a.first()?.as_f64()?, a.get(1)?.as_f64()?, a.get(2)?.as_f64()?])
}

/// A display face's cylinder in part coordinates, when it is a concave one (a hole): a point on
/// its axis, the axis, the radius.
fn bore(face: &Value) -> Option<(V3, V3, f64)> {
    let positions = points(&face["positions"]);
    if positions.is_empty() {
        return None;
    }
    let (point, axis, radius) = match face["surface"]["surface"].as_str() {
        Some("cylinder") => (
            v3(&face["surface"]["axis_point"])?,
            unit(v3(&face["surface"]["axis_direction"])?),
            face["surface"]["radius"].as_f64()?,
        ),
        Some("plane") => return None,
        _ => {
            let f = frames::face_frame(face)?;
            if f.kind != FrameKind::Axis {
                return None;
            }
            (f.origin, unit(f.z), f.radius?)
        }
    };
    if radius <= 0.0 || norm(axis) < 0.5 {
        return None;
    }
    // Every position sits on the cylinder (a fitted frame can mistake a cone or a torus).
    let radial = |p: V3| {
        let d = sub(p, point);
        sub(d, scale(axis, dot(d, axis)))
    };
    if positions
        .iter()
        .any(|p| (norm(radial(*p)) - radius).abs() > radius * 0.02 + 1e-5)
    {
        return None;
    }
    // Concave: the surface's normals point at the axis. The kernel's normals when there are
    // some, else the triangles' winding (outward on a solid).
    let normals = points(&face["normals"]);
    let mut facing = 0.0;
    if normals.len() == positions.len() {
        for (p, n) in positions.iter().zip(&normals) {
            facing += dot(*n, unit(radial(*p)));
        }
    } else {
        let tris: Vec<usize> = face["triangles"]
            .as_array()?
            .iter()
            .filter_map(|v| v.as_u64().map(|n| n as usize))
            .collect();
        for t in tris.chunks_exact(3) {
            let (a, b, c) = (*positions.get(t[0])?, *positions.get(t[1])?, *positions.get(t[2])?);
            let n = cross(sub(b, a), sub(c, a));
            let mid = scale(frames::add(frames::add(a, b), c), 1.0 / 3.0);
            facing += dot(n, unit(radial(mid)));
        }
    }
    (facing < 0.0).then_some((point, axis, radius))
}

/// Every hole of the assembly's display, in assembly coordinates, by occurrence (fastener
/// instances left out).
fn holes(index: &Value, display: &Value) -> Vec<Hole> {
    let fastener_sets: BTreeSet<&str> = index["fasteners"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|f| f["set"].as_str())
        .collect();
    let rows = index["occurrences"].as_array().cloned().unwrap_or_default();
    let mut out: Vec<Hole> = Vec::new();
    for occ in display["occurrences"].as_array().into_iter().flatten() {
        let Some(uuid) = occ["uuid"].as_str() else {
            continue;
        };
        let Some(row) = rows.iter().find(|r| r["uuid"].as_str() == Some(uuid)) else {
            continue;
        };
        if row["instance"]["by"]
            .as_str()
            .is_some_and(|by| fastener_sets.contains(by))
        {
            continue;
        }
        let path = row["path"].as_str().unwrap_or(uuid).to_owned();
        let t: Vec<f64> = occ["transform"]
            .as_array()
            .map(|a| a.iter().filter_map(Value::as_f64).collect())
            .unwrap_or_default();
        let Some(prototype) = occ["prototype"].as_u64() else {
            continue;
        };
        let part = &display["prototypes"][prototype as usize]["part"];
        let start = out.len();
        for body in part["bodies"].as_array().into_iter().flatten() {
            if body["flat_pattern"].as_bool() == Some(true) {
                continue;
            }
            for face in body["faces"].as_array().into_iter().flatten() {
                let Some((point, axis, radius)) = bore(face) else {
                    continue;
                };
                let world = frames::to_world(
                    frames::Frame {
                        kind: FrameKind::Axis,
                        origin: point,
                        z: axis,
                        x: frames::perpendicular(axis),
                        radius: Some(radius),
                    },
                    &t,
                );
                // One direction per axis line, so holes compare the same way round.
                let mut axis = unit(world.z);
                let first = (0..3).find(|&i| axis[i].abs() > 1e-9).unwrap_or(2);
                if axis[first] < 0.0 {
                    axis = scale(axis, -1.0);
                }
                let local = |p: V3| {
                    if t.len() < 12 {
                        return p;
                    }
                    [
                        t[0] * p[0] + t[1] * p[1] + t[2] * p[2] + t[9],
                        t[3] * p[0] + t[4] * p[1] + t[5] * p[2] + t[10],
                        t[6] * p[0] + t[7] * p[1] + t[8] * p[2] + t[11],
                    ]
                };
                let along: Vec<f64> = points(&face["positions"])
                    .into_iter()
                    .map(|p| dot(sub(local(p), world.origin), axis))
                    .collect();
                let from = along.iter().cloned().fold(f64::INFINITY, f64::min);
                let to = along.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let pick = format!(
                    "occurrence:{uuid}/face:{}:{}",
                    body["index"],
                    face["index"]
                );
                let hole = Hole {
                    path: path.clone(),
                    item: format!("occurrence:{uuid}"),
                    faces: vec![pick],
                    point: world.origin,
                    axis,
                    radius,
                    from,
                    to,
                };
                // A cylinder split at its seam: the same axis and radius in this part.
                if let Some(same) = out[start..].iter_mut().find(|h| {
                    dot(h.axis, hole.axis) > PARALLEL
                        && h.off_axis(hole.point) < 1e-6
                        && (h.radius - hole.radius).abs() < 1e-6
                }) {
                    let shift = same.along(hole.point);
                    same.from = same.from.min(hole.from + shift);
                    same.to = same.to.max(hole.to + shift);
                    same.faces.extend(hole.faces);
                } else {
                    out.push(hole);
                }
            }
        }
    }
    out
}

/// Two holes in different parts that meet: the offset between their axes, where they meet.
struct Meeting {
    a: usize,
    b: usize,
    offset: f64,
    /// Where they meet along `a`'s axis.
    at: f64,
}

fn meetings(holes: &[Hole], only: &BTreeSet<String>) -> Vec<Meeting> {
    let mut out = Vec::new();
    for (i, a) in holes.iter().enumerate() {
        for (j, b) in holes.iter().enumerate().skip(i + 1) {
            if a.path == b.path || dot(a.axis, b.axis) < PARALLEL {
                continue;
            }
            if !only.is_empty() && !only.contains(&a.item) && !only.contains(&b.item) {
                continue;
            }
            let offset = a.off_axis(b.point);
            if offset >= a.radius + b.radius {
                continue;
            }
            let shift = a.along(b.point);
            let (b_from, b_to) = (b.from + shift, b.to + shift);
            let gap = (b_from - a.to).max(a.from - b_to);
            if gap > STACK_GAP {
                continue;
            }
            let at = if b_from >= a.to - STACK_GAP {
                a.to
            } else if a.from >= b_to - STACK_GAP {
                a.from
            } else {
                (a.from.max(b_from) + a.to.min(b_to)) / 2.0
            };
            out.push(Meeting {
                a: i,
                b: j,
                offset,
                at,
            });
        }
    }
    out
}

fn mm(m: f64) -> String {
    format!("{:.2} mm", m * 1000.0)
}

fn row(cells: Vec<String>, items: Vec<String>, severity: Option<&'static str>) -> Row {
    Row {
        cells,
        items,
        severity,
        legend: None,
        reference: None,
    }
}

/// Runs hole alignment (`fasteners` false) or the hole/fastener check on the assembly `path`.
pub fn hole_check(
    path: &str,
    text: &str,
    files: &Files,
    configuration: Option<&str>,
    values: &Map<String, Value>,
    fasteners: bool,
) -> Result<Analysis, String> {
    let tolerance = values
        .get("tolerance")
        .and_then(Value::as_f64)
        .unwrap_or(0.1)
        / 1000.0;
    let only: BTreeSet<String> = values
        .get("components")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .map(|s| s.split("/face:").next().unwrap_or(s).to_owned())
        .collect();
    let (index, display, problems) = crate::runtime::compile_assembly(path, text, files)
        .map_err(|d| d.first().map(|d| d.message.clone()).unwrap_or_default())?;
    let holes = holes(&index, &display);
    let mut rows = Vec::new();
    let mut dimensions = Vec::new();
    let found = meetings(&holes, &only);
    let mut misaligned = 0;
    for m in &found {
        let (a, b) = (&holes[m.a], &holes[m.b]);
        // The holes' faces: selecting the row shows the two bores.
        let items: Vec<String> = a.faces.iter().chain(&b.faces).cloned().collect();
        let sizes = if (a.radius - b.radius).abs() > 1e-6 {
            format!(" · Ø{:.2} and Ø{:.2} mm", a.diameter_mm(), b.diameter_mm())
        } else {
            format!(" · Ø{:.2} mm", a.diameter_mm())
        };
        let (finding, severity) = if m.offset > tolerance {
            misaligned += 1;
            let from = a.at(m.at);
            let foot = b.at(b.along(from));
            dimensions.push(ModelDimension {
                from,
                to: foot,
                label: mm(m.offset),
                extension_lines: Vec::new(),
            });
            (format!("Off by {}{sizes}", mm(m.offset)), Some("error"))
        } else {
            (format!("Aligned{sizes}"), None)
        };
        rows.push(row(
            vec![a.path.clone(), b.path.clone(), mm(m.offset), finding],
            items,
            severity,
        ));
    }
    let mut summary = format!(
        "{} hole{} in {} pair{} that meet · {} off by more than {}",
        holes.len(),
        if holes.len() == 1 { "" } else { "s" },
        found.len(),
        if found.len() == 1 { "" } else { "s" },
        misaligned,
        mm(tolerance)
    );
    let mut fastener_problems = 0;
    let mut checked_fasteners = 0;
    if fasteners {
        let resolution = crate::runtime::resolve_assembly_in(path, text, files, configuration)
            .map_err(|d| d.first().map(|d| d.message.clone()).unwrap_or_default())?;
        let resolved = &resolution.resolved;
        for set in &resolved.fasteners {
            let world = resolved.node(set.owner).world;
            let owner = resolved.path(set.owner);
            let e = &set.expansion;
            for k in 0..e.len() {
                checked_fasteners += 1;
                let key = e.keys[k].to_string();
                let label = if owner.is_empty() {
                    format!("{} {key}", e.set)
                } else {
                    format!("{owner}/{} {key}", e.set)
                };
                let p = world.apply(e.positions[k]);
                let axis = unit(world.apply_vector(e.axes[k]));
                let (d, grip) = (e.diameters[k], e.grips[k]);
                // The holes the fastener passes through: parallel, its axis inside the bore.
                let on: Vec<&Hole> = holes
                    .iter()
                    .filter(|h| dot(h.axis, axis).abs() > PARALLEL && h.off_axis(p) < h.radius)
                    .collect();
                let part = e
                    .parts
                    .get(e.selection.get(k).copied().unwrap_or(0) as usize)
                    .map(|p| p.part_number.clone())
                    .unwrap_or_default();
                let mut items: Vec<String> =
                    on.iter().flat_map(|h| h.faces.iter().cloned()).collect();
                items.extend(on.iter().map(|h| h.item.clone()));
                if on.is_empty() {
                    fastener_problems += 1;
                    rows.push(row(
                        vec![label, part, "—".into(), "No hole on its axis".into()],
                        items,
                        Some("warning"),
                    ));
                    continue;
                }
                let mut findings = Vec::new();
                let mut severity = None;
                if let Some(small) = on.iter().find(|h| h.radius * 2.0 < d - 1e-6) {
                    findings.push(format!(
                        "Ø{:.2} mm fastener in the Ø{:.2} mm hole of {}",
                        d * 1000.0,
                        small.diameter_mm(),
                        small.path
                    ));
                    severity = Some("error");
                }
                let worst = on
                    .iter()
                    .map(|h| (h.off_axis(p), h))
                    .fold((0.0, None), |(m, w), (o, h)| {
                        if o > m {
                            (o, Some(h))
                        } else {
                            (m, w)
                        }
                    });
                if let (o, Some(h)) = worst {
                    if o > (h.radius * 2.0 - d).max(0.0) / 2.0 + tolerance {
                        findings.push(format!("{} off {}'s hole axis", mm(o), h.path));
                        severity = Some("error");
                    }
                }
                // The stack: the holes' union along the fastener's axis.
                let mut spans: Vec<(f64, f64)> = on
                    .iter()
                    .map(|h| {
                        let s = dot(sub(h.point, p), axis);
                        let sign = dot(h.axis, axis).signum();
                        let (x, y) = (s + sign * h.from, s + sign * h.to);
                        (x.min(y), x.max(y))
                    })
                    .collect();
                spans.sort_by(|x, y| x.0.total_cmp(&y.0));
                let mut stack = 0.0;
                let mut cur: Option<(f64, f64)> = None;
                for (x, y) in spans {
                    cur = match cur {
                        Some((cx, cy)) if x <= cy + STACK_GAP => Some((cx, cy.max(y))),
                        Some((cx, cy)) => {
                            stack += cy - cx;
                            Some((x, y))
                        }
                        None => Some((x, y)),
                    };
                }
                if let Some((cx, cy)) = cur {
                    stack += cy - cx;
                }
                if grip > 0.0 && (stack - grip).abs() > tolerance.max(0.0005) {
                    findings.push(format!(
                        "Grip {} for a {} stack",
                        mm(grip),
                        mm(stack)
                    ));
                    severity = severity.or(Some("warning"));
                }
                if severity.is_some() {
                    fastener_problems += 1;
                }
                let finding = if findings.is_empty() {
                    format!("Fits · stack {}", mm(stack))
                } else {
                    findings.join(" · ")
                };
                rows.push(row(vec![label, part, mm(stack), finding], items, severity));
            }
        }
        summary += &format!(
            " · {checked_fasteners} fastener{} checked, {fastener_problems} with a problem",
            if checked_fasteners == 1 { "" } else { "s" }
        );
    }
    let data = json!({
        "holes": holes.len(),
        "pairs": found.len(),
        "misaligned": misaligned,
        "fasteners": checked_fasteners,
        "fastener_problems": fastener_problems,
    });
    Ok(Analysis {
        title: if fasteners {
            "Hole/fastener check"
        } else {
            "Hole alignment"
        }
        .into(),
        summary,
        columns: if fasteners {
            ["Hole or fastener", "With", "Offset or stack", "Finding"]
        } else {
            ["Hole", "With", "Offset", "Finding"]
        }
        .map(String::from)
        .to_vec(),
        rows,
        problems: problems
            .iter()
            .filter(|d| d.severity == "error")
            .map(|d| d.message.clone())
            .collect(),
        data,
        settings: vec![format!("Tolerance: {}", mm(tolerance))],
        dimensions,
        ..Default::default()
    })
}
