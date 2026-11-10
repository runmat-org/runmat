//! The `stance` section of `design.inspect`: how an assembly stands on the ground through a set
//! of contact components (a landing gear's skid tubes, a frame's feet).
//!
//! The ground is the plane under the contact set, square to `up` (the assembly's +Z unless
//! given), through the contact set's lowest point. The section reports:
//! - ground clearance: the lowest point of every other component above that plane, lowest
//!   first. A component at or below the plane meets the ground before the contact set does;
//! - support: the convex outline of the contact set's points that touch the plane (within
//!   [`TOUCH_MM`]), and where the centre of mass falls in it. Each edge gives the centre of mass's
//!   margin to it and the tip-over angle about it, `atan(margin / height of the centre of mass)`:
//!   how far the assembly can lean over that edge before it falls;
//! - stance: the support's length and width along the ground axes against the assembly's
//!   overall size.
//!
//! The lowest points are exact on the tessellated bodies: each component's box (the one the
//! interference check keeps, else its display's) screens it, and the lowest few are measured on
//! their tessellation, lowest box first, until no unmeasured box reaches below the lowest point
//! measured. The centre of mass is the Evaluate tab's, from each part's material.

use crate::{Files, ToolError};
use runmat_cad::assembly::model::DefinitionKind;
use runmat_cad::regen::kernel::Transform;
use serde_json::{json, Value};

/// A point of the contact set within this height of its lowest point touches the ground, mm.
pub const TOUCH_MM: f64 = 0.5;
/// The clearance below which a component is flagged, mm.
pub const MIN_CLEARANCE_MM: f64 = 10.0;
/// The tip-over angle below which an edge is flagged, degrees.
pub const MIN_TIP_DEG: f64 = 30.0;
/// The components measured on their tessellation per round (the display cache holds four).
const BATCH: usize = 4;
/// Ground clearance rows listed.
const ROWS: usize = 8;

type V3 = [f64; 3];

fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn cross(a: V3, b: V3) -> V3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
fn unit(a: V3) -> Option<V3> {
    let n = dot(a, a).sqrt();
    (n > 1e-12).then(|| [a[0] / n, a[1] / n, a[2] / n])
}
fn mm(v: f64) -> f64 {
    (v * 1e4).round() / 10.0
}
fn deg(v: f64) -> f64 {
    (v.to_degrees() * 10.0).round() / 10.0
}

/// The ground frame: `up`, and two axes in the ground plane with their names (`x`, `y` when
/// `up` is an assembly axis).
struct Ground {
    up: V3,
    e: [V3; 2],
    names: [String; 2],
}

impl Ground {
    fn new(up: Option<&[f64]>) -> Result<Self, ToolError> {
        let up = match up {
            None => [0.0, 0.0, 1.0],
            Some([x, y, z]) => unit([*x, *y, *z]).ok_or_else(|| ToolError::invalid("up must not be zero"))?,
            Some(_) => return Err(ToolError::invalid("up takes three numbers, [0, 0, 1] for +Z")),
        };
        let axes = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        // The assembly axes most square to `up`, in order, give the ground axes.
        let mut order: Vec<usize> = (0..3).collect();
        order.sort_by(|&a, &b| dot(axes[a], up).abs().total_cmp(&dot(axes[b], up).abs()));
        let first = unit(sub(axes[order[0]], scale(up, dot(axes[order[0]], up)))).unwrap_or(axes[order[0]]);
        let aligned = |v: V3| (0..3).find(|&k| (v[k].abs() - 1.0).abs() < 1e-9);
        // An axis along an assembly axis points along its positive direction, so `+y` reads true.
        let positive = |v: V3| match aligned(v) {
            Some(k) if v[k] < 0.0 => scale(v, -1.0),
            _ => v,
        };
        let first = positive(first);
        let second = positive(cross(up, first));
        let name = |v: V3, fallback: &str| match aligned(v) {
            Some(k) => ["x", "y", "z"][k].to_owned(),
            None => fallback.to_owned(),
        };
        // Keep the lower-numbered assembly axis first (x before y).
        let (a, b) = if aligned(second).zip(aligned(first)).is_some_and(|(s, f)| s < f) {
            (second, first)
        } else {
            (first, second)
        };
        Ok(Self { up, e: [a, b], names: [name(a, "u"), name(b, "v")] })
    }
    fn flat(&self, p: V3) -> [f64; 2] {
        [dot(p, self.e[0]), dot(p, self.e[1])]
    }
    /// An outward direction in the ground plane as the nearest signed axis (`+y`).
    fn side(&self, n: [f64; 2]) -> String {
        let k = usize::from(n[1].abs() > n[0].abs());
        format!("{}{}", if n[k] >= 0.0 { "+" } else { "-" }, self.names[k])
    }
}

fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn scale(a: V3, s: f64) -> V3 {
    [a[0] * s, a[1] * s, a[2] * s]
}

/// One part or board occurrence: its world box corners and, once measured, its tessellation's
/// points in the world.
struct Occurrence {
    path: String,
    part: bool,
    corners: Option<Vec<V3>>,
    points: Option<Vec<V3>>,
}

impl Occurrence {
    fn box_low(&self, g: &Ground) -> f64 {
        self.corners.as_ref().map_or(f64::INFINITY, |c| c.iter().map(|p| dot(*p, g.up)).fold(f64::INFINITY, f64::min))
    }
    fn low(&self, g: &Ground) -> f64 {
        match &self.points {
            Some(p) => p.iter().map(|p| dot(*p, g.up)).fold(f64::INFINITY, f64::min),
            None => self.box_low(g),
        }
    }
}

fn corners(min: V3, max: V3, world: &Transform) -> Vec<V3> {
    (0..8)
        .map(|k| {
            world.apply([
                if k & 1 == 0 { min[0] } else { max[0] },
                if k & 2 == 0 { min[1] } else { max[1] },
                if k & 4 == 0 { min[2] } else { max[2] },
            ])
        })
        .collect()
}

/// Measures the named occurrences on their tessellation (one resolution of the assembly).
fn measure(path: &str, text: &str, files: &Files, all: &mut [Occurrence], names: &[String]) -> Result<(), ToolError> {
    if names.is_empty() {
        return Ok(());
    }
    let placed = runmat_cad_wasm::probe::placed_parts(path, text, files, names)
        .map_err(|e| ToolError::new("not_compiled", format!("{path} doesn't resolve: {e}")))?;
    for p in placed {
        let Some(o) = all.iter_mut().find(|o| o.path == p.occurrence) else { continue };
        let mut points = Vec::new();
        let (mut lo, mut hi) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
        for face in p.display.bodies.iter().filter(|b| !b.flat_pattern).flat_map(|b| b.faces.iter()) {
            for q in face.positions.chunks_exact(3) {
                let local = [q[0] as f64, q[1] as f64, q[2] as f64];
                for k in 0..3 {
                    lo[k] = lo[k].min(local[k]);
                    hi[k] = hi[k].max(local[k]);
                }
                points.push(p.world.apply(local));
            }
        }
        if lo[0] <= hi[0] {
            o.corners = Some(corners(lo, hi, &p.world));
        }
        o.points = Some(points);
    }
    Ok(())
}

/// The convex hull of `points`, counter-clockwise (Andrew's monotone chain).
fn hull(mut points: Vec<[f64; 2]>) -> Vec<[f64; 2]> {
    points.sort_by(|a, b| a[0].total_cmp(&b[0]).then(a[1].total_cmp(&b[1])));
    points.dedup_by(|a, b| (a[0] - b[0]).abs() < 1e-9 && (a[1] - b[1]).abs() < 1e-9);
    if points.len() < 3 {
        return points;
    }
    let turn = |o: [f64; 2], a: [f64; 2], b: [f64; 2]| (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
    let mut lower: Vec<[f64; 2]> = Vec::new();
    for &p in &points {
        while lower.len() >= 2 && turn(lower[lower.len() - 2], lower[lower.len() - 1], p) <= 1e-12 {
            lower.pop();
        }
        lower.push(p);
    }
    let mut upper: Vec<[f64; 2]> = Vec::new();
    for &p in points.iter().rev() {
        while upper.len() >= 2 && turn(upper[upper.len() - 2], upper[upper.len() - 1], p) <= 1e-12 {
            upper.pop();
        }
        upper.push(p);
    }
    lower.pop();
    upper.pop();
    lower.extend(upper);
    lower
}

/// The distance from `c` to the segment `a`–`b`.
fn to_segment(c: [f64; 2], a: [f64; 2], b: [f64; 2]) -> f64 {
    let d = [b[0] - a[0], b[1] - a[1]];
    let len2 = d[0] * d[0] + d[1] * d[1];
    let t = if len2 > 0.0 { (((c[0] - a[0]) * d[0] + (c[1] - a[1]) * d[1]) / len2).clamp(0.0, 1.0) } else { 0.0 };
    ((c[0] - a[0] - t * d[0]).powi(2) + (c[1] - a[1] - t * d[1]).powi(2)).sqrt()
}

pub(super) fn stance(path: &str, text: &str, files: &Files, contact: Option<&[String]>, up: Option<&[f64]>) -> Result<Value, ToolError> {
    let contact = contact.filter(|c| !c.is_empty()).ok_or_else(|| {
        ToolError::invalid("the stance section needs contact: the components that touch the ground (the skid tubes, the feet)")
    })?;
    let g = Ground::new(up)?;
    let resolution = runmat_cad_wasm::runtime::resolve_assembly(path, text, files)
        .map_err(|d| ToolError::new("not_compiled", format!("{path} doesn't resolve: {}", d.first().map(|d| d.message.as_str()).unwrap_or(""))))?;
    let r = &resolution.resolved;
    let mut notes = Vec::new();
    let mut all = Vec::new();
    for (id, node) in r.nodes.iter().enumerate() {
        if node.parent.is_none() || node.failed || node.represented_by_ancestor {
            continue;
        }
        let part = match node.definition.kind {
            DefinitionKind::Part => true,
            DefinitionKind::Circuit => false,
            _ => continue,
        };
        let key = runmat_cad_wasm::evaluate::content_key(&node.definition.file.0, &node.configuration.digest(), files);
        let corners = key.and_then(runmat_cad_wasm::evaluate::cached_bbox).filter(|b| !b.is_empty()).map(|b| corners(b.min, b.max, &node.world));
        all.push(Occurrence { path: r.path(id as u32), part, corners, points: None });
    }
    let paths: Vec<String> = all.iter().map(|o| o.path.clone()).collect();
    let contact = super::surface::occurrences_named(contact, &paths)?;
    // Parts without a kept box, and the contact set, are measured on their tessellation.
    let first: Vec<String> = all.iter().filter(|o| o.part && (o.corners.is_none() || contact.contains(&o.path))).map(|o| o.path.clone()).collect();
    measure(path, text, files, &mut all, &first)?;
    for o in all.iter().filter(|o| o.corners.is_none()) {
        notes.push(format!("{} has no box yet (run the interference check once): its clearance isn't measured", o.path));
    }
    let contact_points: Vec<V3> = all.iter().filter(|o| contact.contains(&o.path)).flat_map(|o| o.points.clone().unwrap_or_default()).collect();
    if contact_points.is_empty() {
        return Err(ToolError::invalid(format!("the contact set {} has no part body to stand on", contact.join(", "))));
    }
    let ground = contact_points.iter().map(|p| dot(*p, g.up)).fold(f64::INFINITY, f64::min);

    // Ground clearance: measure the others lowest box first until no unmeasured box reaches
    // below the lowest point measured.
    loop {
        let lowest = all.iter().filter(|o| !contact.contains(&o.path) && o.points.is_some()).map(|o| o.low(&g)).fold(f64::INFINITY, f64::min);
        let mut next: Vec<&Occurrence> = all.iter().filter(|o| o.part && o.points.is_none() && !contact.contains(&o.path) && o.box_low(&g) < lowest).collect();
        if next.is_empty() {
            break;
        }
        next.sort_by(|a, b| a.box_low(&g).total_cmp(&b.box_low(&g)));
        let names: Vec<String> = next.iter().take(BATCH).map(|o| o.path.clone()).collect();
        measure(path, text, files, &mut all, &names)?;
        // A part that didn't regenerate keeps its box.
        for o in all.iter_mut().filter(|o| names.contains(&o.path) && o.points.is_none()) {
            o.points = o.corners.clone();
        }
    }
    let mut rows: Vec<(f64, &Occurrence)> = all.iter().filter(|o| !contact.contains(&o.path) && o.corners.is_some()).map(|o| (o.low(&g) - ground, o)).collect();
    rows.sort_by(|a, b| a.0.total_cmp(&b.0));
    let clearance_rows: Vec<Value> = rows
        .iter()
        .take(ROWS)
        .map(|(c, o)| {
            let mut row = json!({ "component": o.path, "clearance_mm": mm(*c) });
            if o.points.is_none() {
                // Only its box was measured: its lowest point is at least this high.
                row["at_least"] = json!(true);
            }
            row
        })
        .collect();
    let lowest = rows.first().map(|(c, o)| (*c, o.path.clone()));

    // The centre of mass, from each part's material.
    let mass = runmat_cad_wasm::mass::assembly_mass(path, text, files, None).map_err(|e| ToolError::new("not_compiled", format!("mass properties: {e}")))?;
    notes.extend(mass.problems.iter().map(|p| format!("mass: {p}")));
    let com = mass.total.center;
    let height = dot(com, g.up) - ground;
    let c = g.flat(com);

    // The support: the contact points on the ground, as a convex outline.
    let touching: Vec<[f64; 2]> = contact_points.iter().filter(|p| dot(**p, g.up) <= ground + TOUCH_MM / 1000.0).map(|p| g.flat(*p)).collect();
    let outline = hull(touching);
    let area = (0..outline.len()).map(|i| {
        let (a, b) = (outline[i], outline[(i + 1) % outline.len()]);
        a[0] * b[1] - b[0] * a[1]
    }).sum::<f64>() / 2.0;
    let mut edges = Vec::new();
    let (inside, margin) = if outline.len() >= 3 && area > 1e-6 {
        let mut least = f64::INFINITY;
        for i in 0..outline.len() {
            let (a, b) = (outline[i], outline[(i + 1) % outline.len()]);
            let d = [b[0] - a[0], b[1] - a[1]];
            let len = (d[0] * d[0] + d[1] * d[1]).sqrt();
            if len < 1e-9 {
                continue;
            }
            // Counter-clockwise: the outward normal is the edge turned clockwise.
            let n = [d[1] / len, -d[0] / len];
            let m = (a[0] - c[0]) * n[0] + (a[1] - c[1]) * n[1];
            least = least.min(m);
            edges.push((m, json!({
                "side": g.side(n),
                "length_mm": mm(len),
                "margin_mm": mm(m),
                "tip_angle_deg": deg(m.atan2(height.max(1e-9))),
            })));
        }
        (least > 0.0, least)
    } else {
        // A line or a point holds nothing up across it.
        let least = match outline.len() {
            0 => f64::INFINITY,
            1 => to_segment(c, outline[0], outline[0]),
            _ => (0..outline.len()).map(|i| to_segment(c, outline[i], outline[(i + 1) % outline.len()])).fold(f64::INFINITY, f64::min),
        };
        (false, -least)
    };
    edges.sort_by(|a, b| a.0.total_cmp(&b.0));
    let tip = inside.then(|| margin.atan2(height.max(1e-9)));
    let least_side = edges.first().map(|e| e.1["side"].as_str().unwrap_or("").to_owned());

    // Stance against the overall size (every component's box).
    let extent = |pts: &mut dyn Iterator<Item = [f64; 2]>| {
        let (mut lo, mut hi) = ([f64::INFINITY; 2], [f64::NEG_INFINITY; 2]);
        for p in pts {
            for k in 0..2 {
                lo[k] = lo[k].min(p[k]);
                hi[k] = hi[k].max(p[k]);
            }
        }
        [hi[0] - lo[0], hi[1] - lo[1]]
    };
    let support_size = extent(&mut outline.iter().copied());
    let box_points: Vec<V3> = all.iter().flat_map(|o| o.corners.clone().unwrap_or_default()).collect();
    let overall = extent(&mut box_points.iter().map(|p| g.flat(*p)));
    let overall_height = box_points.iter().map(|p| dot(*p, g.up)).fold(f64::NEG_INFINITY, f64::max) - box_points.iter().map(|p| dot(*p, g.up)).fold(f64::INFINITY, f64::min);
    let [a, b] = &g.names;
    let ratio = |s: f64, o: f64| if o > 0.0 { (s / o * 100.0).round() / 100.0 } else { 0.0 };

    // Findings.
    let mut findings = Vec::new();
    if let Some((clear, name)) = &lowest {
        if *clear <= 0.0 {
            findings.push(json!({ "severity": "error", "message": format!("{name} reaches {} mm below the contact set's lowest point, so it touches the ground first", mm(-clear)) }));
        } else if *clear * 1000.0 < MIN_CLEARANCE_MM {
            findings.push(json!({ "severity": "warning", "message": format!("{name} clears the ground by only {} mm (at least {MIN_CLEARANCE_MM} mm)", mm(*clear)) }));
        }
    }
    if !inside {
        let side = least_side.clone().map(|s| format!(" past its {s} edge")).unwrap_or_default();
        findings.push(json!({ "severity": "error", "message": if outline.len() < 3 || area <= 1e-6 {
            format!("the contact set touches the ground along a line or at a point, so the assembly tips over; the centre of mass is {} mm off it", mm(-margin))
        } else {
            format!("the centre of mass falls {} mm outside the support{side}, so the assembly tips over", mm(-margin))
        } }));
    } else if let Some(t) = tip.filter(|t| t.to_degrees() < MIN_TIP_DEG) {
        findings.push(json!({ "severity": "warning", "message": format!("it tips over its {} edge after leaning {}° (at least {MIN_TIP_DEG}°): widen or lengthen the support there", least_side.clone().unwrap_or_default(), deg(t)) }));
    }
    let ok = findings.iter().all(|f| f["severity"] != "error");

    let clearance_text = match &lowest {
        Some((clear, name)) => format!("lowest other component {name}, {} mm above the ground", mm(*clear)),
        None => "no other component".to_owned(),
    };
    let support_text = if inside {
        format!(
            "centre of mass inside the support, {} mm from its {} edge (tips at {}°)",
            mm(margin),
            least_side.clone().unwrap_or_default(),
            tip.map_or(0.0, deg)
        )
    } else {
        format!("centre of mass {} mm outside the support", mm(-margin))
    };
    let summary = format!(
        "Standing on {}: {clearance_text}; {support_text}; stance {} × {} mm ({a} × {b}) against {} × {} mm overall.",
        contact.join(", "),
        mm(support_size[0]),
        mm(support_size[1]),
        mm(overall[0]),
        mm(overall[1]),
    );
    let mut value = json!({
        "summary": summary,
        "ok": ok,
        "findings": findings,
        "contact": contact,
        "up": g.up,
        "ground_mm": mm(ground),
        "ground_clearance": {
            "lowest": lowest.as_ref().map(|(c, n)| json!({ "component": n, "clearance_mm": mm(*c) })),
            "components": clearance_rows,
            "required_mm": MIN_CLEARANCE_MM,
        },
        "center_of_mass": {
            "point_mm": [mm(com[0]), mm(com[1]), mm(com[2])],
            "height_mm": mm(height),
            "mass_kg": (mass.total.mass * 1e4).round() / 1e4,
            "complete": mass.complete(),
        },
        "support": {
            "corners_mm": outline.iter().map(|p| [mm(p[0]), mm(p[1])]).collect::<Vec<_>>(),
            "com_inside": inside,
            "min_margin_mm": mm(margin),
            "min_tip_angle_deg": tip.map(deg),
            "required_tip_angle_deg": MIN_TIP_DEG,
            "edges": edges.into_iter().map(|e| e.1).collect::<Vec<_>>(),
        },
        "stance": {
            format!("{a}_mm"): mm(support_size[0]),
            format!("{b}_mm"): mm(support_size[1]),
            "overall_mm": { a.clone(): mm(overall[0]), b.clone(): mm(overall[1]), "height": mm(overall_height) },
            "ratio": { a.clone(): ratio(support_size[0], overall[0]), b.clone(): ratio(support_size[1], overall[1]) },
        },
    });
    if !notes.is_empty() {
        value["notes"] = json!(notes);
    }
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_square_hull_drops_its_inner_points() {
        let h = hull(vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.5, 0.5], [0.5, 0.0]]);
        assert_eq!(h, vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]);
    }

    #[test]
    fn ground_axes_follow_up() {
        let z = Ground::new(None).unwrap();
        assert_eq!(z.names, ["x".to_owned(), "y".to_owned()]);
        assert_eq!(z.side([0.0, -1.0]), "-y");
        let y = Ground::new(Some(&[0.0, 1.0, 0.0])).unwrap();
        assert_eq!(y.names, ["x".to_owned(), "z".to_owned()]);
    }
}
