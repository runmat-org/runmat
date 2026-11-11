//! `view.measure` on a part or an assembly: what Measure's form shows and what its pin leaves on
//! the model. The browser, the Desktop, the agent and the CLI all run this one implementation
//! through [`crate::analyze`].
//!
//! `values.items` holds the picks ([`items`]); `values.mode` is one of:
//!
//! - `distance` (the default). Two items give their minimum distance and its X, Y and Z parts,
//!   on the display tessellation (within its chordal tolerance of the exact surfaces). Two
//!   parallel planar faces also give their normal distance, two parallel axes (cylinders,
//!   circular edges) the distance between the axes, and two items with directions their angle.
//!   Bodies and components measure their nearest material. One item gives its own size: an
//!   edge's length (an arc's exact length), a face's area, a circle's or cylinder's radius and
//!   diameter, a vertex's position, a body's extent.
//! - `angle`: two planar faces, straight edges or axes, in any combination.
//! - `radius`: each circular edge's or cylindrical face's radius and diameter.
//! - `path`: the edges' summed length.
//!
//! Results come back as rows (`Quantity`, `Value`) and as `dimensions`: lines between two model
//! points with a label, which the Desktop draws while the results are open and keeps on the
//! model when the result is pinned.

mod distance;
mod items;
#[cfg(test)]
mod tests;

use crate::analyze::{Analysis, ModelDimension, Row};
use crate::edit::assembly::frames::{add, dot, norm, perpendicular, scale, sub, V3};
use crate::features::picks_of;
use crate::runtime::Files;
use distance::{nearest, Nearest};
use items::{bodies_item, on_part, Circle, Direction, Item, Kind, Placed};
use runmat_cad::regen::kernel::{TessellationOptions, Transform};
use serde_json::{json, Map, Value};

pub use distance::Primitives;

fn identity() -> Transform {
    Transform::translation([0.0; 3])
}

fn mm(m: f64) -> String {
    format!("{:.3} mm", m * 1000.0)
}

fn mm2(m2: f64) -> String {
    format!("{:.3} mm²", m2 * 1e6)
}

fn deg(r: f64) -> String {
    format!("{:.2}°", r.to_degrees())
}

/// Parallel within about a hundredth of a degree.
const PARALLEL: f64 = 0.999_999_98;

/// The picks of a part file.
fn part_items(path: &str, text: &str, configuration: Option<&str>, picks: &[String]) -> Result<Vec<Item>, String> {
    let part = crate::runtime::regenerate_part_in(path, text, configuration, &TessellationOptions::default())
        .map_err(|d| d.first().map(|d| d.message.clone()).unwrap_or_else(|| "the part doesn't regenerate".into()))?;
    let at = Placed { display: &part.display, world: identity(), occurrence: String::new() };
    picks
        .iter()
        .map(|pick| on_part(&at, pick, pick).ok_or_else(|| format!("`{pick}` isn't a face, edge, vertex or body of this part")))
        .collect()
}

/// The picks of an assembly: components, or faces, edges and vertices of one.
fn assembly_items(path: &str, text: &str, files: &Files, picks: &[String]) -> Result<Vec<Item>, String> {
    let (index, _, _) = crate::runtime::compile_assembly(path, text, files).map_err(|d| {
        d.first().map(|d| d.message.clone()).unwrap_or_else(|| "the assembly doesn't compile".into())
    })?;
    let rows = index["occurrences"].as_array().cloned().unwrap_or_default();
    let path_of = |uuid: &str| -> Result<String, String> {
        rows.iter()
            .find(|r| r["uuid"].as_str() == Some(uuid))
            .and_then(|r| r["path"].as_str().map(str::to_owned))
            .ok_or_else(|| "a picked component isn't in this assembly any more".to_string())
    };
    let mut wanted = Vec::new();
    let mut parsed = Vec::new();
    for pick in picks {
        let rest = pick.strip_prefix("occurrence:").ok_or_else(|| format!("`{pick}` isn't a component or an entity of one"))?;
        let (uuid, entity) = match rest.split_once('/') {
            Some((u, e)) => (u, Some(e.to_owned())),
            None => (rest, None),
        };
        let occurrence = path_of(uuid)?;
        wanted.push(occurrence.clone());
        parsed.push((pick.clone(), occurrence, entity));
    }
    let placed = crate::probe::placed_parts(path, text, files, &wanted)?;
    parsed
        .into_iter()
        .map(|(pick, occurrence, entity)| {
            let under: Vec<Placed<'_>> = placed
                .iter()
                .filter(|p| p.occurrence == occurrence || p.occurrence.starts_with(&format!("{occurrence}/")))
                .map(|p| Placed { display: &p.display, world: p.world, occurrence: p.occurrence.clone() })
                .collect();
            match entity {
                None => bodies_item(&under, &pick, occurrence.clone(), None)
                    .ok_or_else(|| format!("`{occurrence}` has no bodies to measure")),
                Some(entity) => {
                    let at = under
                        .into_iter()
                        .find(|p| p.occurrence == occurrence)
                        .ok_or_else(|| format!("`{occurrence}` isn't drawn, so its faces can't be measured"))?;
                    on_part(&at, &pick, &entity).ok_or_else(|| format!("the picked entity isn't on `{occurrence}` any more"))
                }
            }
        })
        .collect()
}

fn row(quantity: &str, value: String) -> Row {
    Row { cells: vec![quantity.into(), value], items: Vec::new(), severity: None, legend: None, reference: None }
}

fn line(from: V3, to: V3, label: String) -> ModelDimension {
    ModelDimension { from, to, label }
}

/// The angle between two directions, as people read it: between lines or planes 0° to 90°,
/// between a line and a plane its tilt out of the plane. A circular edge reads as its plane.
pub fn angle(a: &Direction, b: &Direction) -> f64 {
    let (u, line_u) = along(a);
    let (v, line_v) = along(b);
    let c = dot(u, v).abs().min(1.0);
    if line_u == line_v {
        c.acos()
    } else {
        std::f64::consts::FRAC_PI_2 - c.acos()
    }
}

/// The direction's vector and whether it is a line (true) or a plane's normal (false).
fn along(d: &Direction) -> (V3, bool) {
    match *d {
        Direction::Line(_, v) | Direction::Axis(_, v) => (v, true),
        Direction::Plane(_, n) => (n, false),
    }
}

fn point_of(d: &Direction) -> V3 {
    match *d {
        Direction::Line(p, _) | Direction::Axis(p, _) | Direction::Plane(p, _) => p,
    }
}

/// The normal distance of two parallel planes, with the foot of the first's point on the second.
fn plane_gap(a: &Direction, b: &Direction) -> Option<(f64, V3, V3)> {
    let (Direction::Plane(p, n), Direction::Plane(q, m)) = (*a, *b) else { return None };
    (dot(n, m).abs() >= PARALLEL).then(|| {
        let d = dot(sub(p, q), m);
        (d.abs(), p, sub(p, scale(m, d)))
    })
}

/// The distance between two parallel axes (cylinders, circular edges), with the foot of the
/// first's centre on the second.
fn axis_gap(a: &Circle, b: &Circle) -> Option<(f64, V3, V3)> {
    let (p, u, q, v) = (a.centre, a.axis, b.centre, b.axis);
    (dot(u, v).abs() >= PARALLEL).then(|| {
        let foot = add(q, scale(v, dot(sub(p, q), v)));
        (norm(sub(p, foot)), p, foot)
    })
}

fn circle_rows(c: &Circle, rows: &mut Vec<Row>, dims: &mut Vec<ModelDimension>) {
    rows.push(row("Radius", mm(c.radius)));
    rows.push(row("Diameter", mm(2.0 * c.radius)));
    let rim = add(c.centre, scale(perpendicular(c.axis), c.radius));
    dims.push(line(c.centre, rim, format!("R {}", mm(c.radius))));
}

/// One item's own size.
fn single(item: &Item, rows: &mut Vec<Row>, dims: &mut Vec<ModelDimension>) -> String {
    match item.kind {
        Kind::Vertex => {
            let p = item.anchor;
            rows.push(row("X", mm(p[0])));
            rows.push(row("Y", mm(p[1])));
            rows.push(row("Z", mm(p[2])));
            format!("Position ({:.3}, {:.3}, {:.3}) mm", p[0] * 1e3, p[1] * 1e3, p[2] * 1e3)
        }
        Kind::Edge => {
            let length = item.length.unwrap_or(0.0);
            rows.push(row("Length", mm(length)));
            match &item.circle {
                Some(c) => {
                    circle_rows(c, rows, dims);
                    format!("Length {} · radius {}", mm(length), mm(c.radius))
                }
                None => format!("Length {}", mm(length)),
            }
        }
        Kind::Face => {
            let area = item.area.unwrap_or(0.0);
            rows.push(row("Area", mm2(area)));
            match &item.circle {
                Some(c) => {
                    circle_rows(c, rows, dims);
                    format!("Area {} · radius {}", mm2(area), mm(c.radius))
                }
                None => format!("Area {}", mm2(area)),
            }
        }
        Kind::Body => {
            let (mut lo, mut hi) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
            for c in item.prims.triangles.iter().flatten() {
                for i in 0..3 {
                    lo[i] = lo[i].min(c[i]);
                    hi[i] = hi[i].max(c[i]);
                }
            }
            for (i, axis) in ["X", "Y", "Z"].iter().enumerate() {
                rows.push(row(&format!("Extent {axis}"), mm(hi[i] - lo[i])));
            }
            format!("Extent {:.3} × {:.3} × {:.3} mm", (hi[0] - lo[0]) * 1e3, (hi[1] - lo[1]) * 1e3, (hi[2] - lo[2]) * 1e3)
        }
    }
}

/// Two items' minimum distance, and what their shapes add.
fn pair(a: &Item, b: &Item, rows: &mut Vec<Row>, dims: &mut Vec<ModelDimension>) -> Result<String, String> {
    let n: Nearest = nearest(&a.prims, &b.prims).ok_or("a picked item has no geometry to measure")?;
    rows.push(row("Minimum distance", mm(n.distance)));
    let d = sub(n.b, n.a);
    rows.push(row("ΔX", mm(d[0].abs())));
    rows.push(row("ΔY", mm(d[1].abs())));
    rows.push(row("ΔZ", mm(d[2].abs())));
    let mut summary = format!("Minimum distance {}", mm(n.distance));
    let planes = match (&a.direction, &b.direction) {
        (Some(x), Some(y)) => plane_gap(x, y).map(|g| ("Normal distance", g)),
        _ => None,
    };
    let axes = match (&a.circle, &b.circle) {
        (Some(x), Some(y)) => axis_gap(x, y).map(|g| ("Axis distance", g)),
        _ => None,
    };
    // Two parallel circles report both: how far apart their planes are and their axes.
    if let (Some(_), Some((name, (g, ..)))) = (planes, axes) {
        rows.push(row(name, mm(g)));
    }
    let gap = planes.or(axes);
    match gap {
        Some((name, (g, from, to))) => {
            rows.push(row(name, mm(g)));
            summary += &format!(" · {} {}", name.to_lowercase(), mm(g));
            dims.push(line(from, to, mm(g)));
        }
        None => dims.push(line(n.a, n.b, mm(n.distance))),
    }
    if let (Some(x), Some(y)) = (&a.direction, &b.direction) {
        let t = angle(x, y);
        if t.to_degrees() > 0.01 {
            rows.push(row("Angle", deg(t)));
            summary += &format!(" · angle {}", deg(t));
        }
    }
    Ok(summary)
}

fn angle_of(a: &Item, b: &Item, rows: &mut Vec<Row>, dims: &mut Vec<ModelDimension>) -> Result<String, String> {
    let need = |i: &Item| {
        i.direction.ok_or_else(|| format!("{} has no direction: an angle takes planar faces, straight edges or axes", i.label))
    };
    let (x, y) = (need(a)?, need(b)?);
    let t = angle(&x, &y);
    rows.push(row("Angle", deg(t)));
    rows.push(row("Supplement", deg(std::f64::consts::PI - t)));
    dims.push(line(point_of(&x), point_of(&y), deg(t)));
    Ok(format!("Angle {}", deg(t)))
}

/// Runs `view.measure`.
pub fn measure(path: &str, text: &str, files: &Files, values: &Map<String, Value>) -> Result<Analysis, String> {
    let configuration = values.get("configuration").and_then(Value::as_str).filter(|c| !c.is_empty() && *c != "default");
    let picks = picks_of(values, "items");
    if picks.is_empty() {
        return Err("pick one or two items to measure: faces, edges, vertices, bodies or components".into());
    }
    let items = if path.ends_with(".assembly") {
        assembly_items(path, text, files, &picks)?
    } else {
        part_items(path, text, configuration, &picks)?
    };
    let mode = values.get("mode").and_then(Value::as_str).unwrap_or("distance");
    let (mut rows, mut dims) = (Vec::new(), Vec::new());
    let summary = match (mode, items.as_slice()) {
        ("distance", [one]) => single(one, &mut rows, &mut dims),
        ("distance", [a, b]) => pair(a, b, &mut rows, &mut dims)?,
        ("distance", _) => return Err("a distance runs between two items; pick one or two".into()),
        ("angle", [a, b]) => angle_of(a, b, &mut rows, &mut dims)?,
        ("angle", _) => return Err("an angle runs between two items: planar faces, straight edges or axes".into()),
        ("radius", all) => {
            for item in all {
                let c = item.circle.ok_or_else(|| format!("{} isn't circular: radius takes circular edges and cylindrical faces", item.label))?;
                if all.len() > 1 {
                    rows.push(row(&item.label, String::new()));
                }
                circle_rows(&c, &mut rows, &mut dims);
            }
            let first = all[0].circle.map(|c| c.radius).unwrap_or(0.0);
            format!("Radius {} · diameter {}", mm(first), mm(2.0 * first))
        }
        ("path", all) => {
            let mut total = 0.0;
            for item in all {
                let l = item.length.ok_or_else(|| format!("{} isn't an edge: path length adds edges", item.label))?;
                total += l;
            }
            rows.push(row("Path length", mm(total)));
            format!("Path length {} · {} edge{}", mm(total), all.len(), if all.len() == 1 { "" } else { "s" })
        }
        (other, _) => return Err(format!("`{other}` isn't a measure: distance, angle, radius or path")),
    };
    let named: Vec<String> = items.iter().map(|i| i.label.clone()).collect();
    let data = json!({
        "mode": mode,
        "items": items.iter().map(|i| json!({ "id": i.id, "label": i.label })).collect::<Vec<_>>(),
        "rows": rows.iter().map(|r| json!({ "quantity": r.cells[0], "value": r.cells[1] })).collect::<Vec<_>>(),
    });
    Ok(Analysis {
        title: "Measure".into(),
        summary,
        columns: ["Quantity", "Value"].map(String::from).to_vec(),
        rows,
        problems: Vec::new(),
        data,
        colors: None,
        settings: named,
        arrow: None,
        dimensions: dims,
    })
}
