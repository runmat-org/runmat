//! Ray queries on a part or an assembly: where a line from a point along a direction meets the
//! bodies, with the face it meets and the surface's normal there. A part seated on a curved
//! body (a clamp foot on a lofted shell) is placed from this point and normal.
//!
//! Each body is tessellated (the display tessellation, part coordinates) and placed at its
//! world transform; the ray is tested against every triangle of the bodies whose world box it
//! passes through. The point lies on the tessellation, within its chordal tolerance of the
//! exact surface, and the normal is the surface's own normal interpolated across the triangle
//! where the kernel gives one (the triangle's normal otherwise).

use crate::configured::regenerate_configured;
use crate::runtime::{regenerate_part_in, resolve_assembly_in, Files};
use runmat_cad::assembly::model::DefinitionKind;
use runmat_cad::display::{part_display, PartDisplay};
use runmat_cad::regen::kernel::{TessellationOptions, Transform};
use serde::Serialize;
use std::collections::BTreeMap;

/// Where the ray meets a body.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Hit {
    /// The occurrence's path in the assembly; empty for a part file.
    pub occurrence: String,
    /// The face's name for people (`Loft side 1`), when it has one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub face: Option<String>,
    /// The face's reference text, when its name has one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reference: Option<String>,
    /// The point, world coordinates (m).
    pub point: [f64; 3],
    /// The surface's outward unit normal there, world coordinates.
    pub normal: [f64; 3],
    /// Distance from the ray's start along the direction (m).
    pub distance: f64,
    /// The ray enters the body here (the normal faces against the direction).
    pub entering: bool,
}

type V3 = [f64; 3];

fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn cross(a: V3, b: V3) -> V3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
fn unit(a: V3) -> Option<V3> {
    let n = dot(a, a).sqrt();
    (n > 1e-300).then(|| [a[0] / n, a[1] / n, a[2] / n])
}

/// The ray's distance to triangle `(a, b, c)` and the barycentric weights of `b` and `c`
/// (Möller–Trumbore), when it meets it ahead of the start.
fn meets(from: V3, dir: V3, a: V3, b: V3, c: V3) -> Option<(f64, f64, f64)> {
    let (e1, e2) = (sub(b, a), sub(c, a));
    let p = cross(dir, e2);
    let det = dot(e1, p);
    if det.abs() < 1e-18 {
        return None;
    }
    let inv = 1.0 / det;
    let s = sub(from, a);
    let u = dot(s, p) * inv;
    if !(-1e-9..=1.0 + 1e-9).contains(&u) {
        return None;
    }
    let q = cross(s, e1);
    let v = dot(dir, q) * inv;
    if v < -1e-9 || u + v > 1.0 + 1e-9 {
        return None;
    }
    let t = dot(e2, q) * inv;
    (t >= 0.0).then_some((t, u, v))
}

/// Whether the ray passes through the box (slab test).
fn passes(from: V3, dir: V3, min: V3, max: V3) -> bool {
    let (mut lo, mut hi) = (0.0f64, f64::INFINITY);
    for i in 0..3 {
        if dir[i].abs() < 1e-300 {
            if from[i] < min[i] || from[i] > max[i] {
                return false;
            }
            continue;
        }
        let (a, b) = ((min[i] - from[i]) / dir[i], (max[i] - from[i]) / dir[i]);
        lo = lo.max(a.min(b));
        hi = hi.min(a.max(b));
    }
    lo <= hi
}

/// The hits of the ray on `display` placed at `world`, appended to `out`.
fn hits_on(display: &PartDisplay, world: &Transform, occurrence: &str, from: V3, dir: V3, out: &mut Vec<Hit>) {
    // The ray in the part's frame, so the tessellation is read as it is.
    let local = world.inverse();
    let (f, d) = (local.apply(from), local.apply_vector(dir));
    for body in display.bodies.iter().filter(|b| !b.flat_pattern) {
        for face in &body.faces {
            let at = |i: u32| -> V3 {
                let k = i as usize * 3;
                [face.positions[k] as f64, face.positions[k + 1] as f64, face.positions[k + 2] as f64]
            };
            let normal_at = |i: u32| -> Option<V3> {
                let k = i as usize * 3;
                face.normals.get(k..k + 3).map(|n| [n[0] as f64, n[1] as f64, n[2] as f64])
            };
            for tri in face.triangles.chunks_exact(3) {
                let (a, b, c) = (at(tri[0]), at(tri[1]), at(tri[2]));
                let Some((t, u, v)) = meets(f, d, a, b, c) else { continue };
                let w = 1.0 - u - v;
                let smooth = match (normal_at(tri[0]), normal_at(tri[1]), normal_at(tri[2])) {
                    (Some(na), Some(nb), Some(nc)) => unit([
                        w * na[0] + u * nb[0] + v * nc[0],
                        w * na[1] + u * nb[1] + v * nc[1],
                        w * na[2] + u * nb[2] + v * nc[2],
                    ]),
                    _ => None,
                };
                let Some(normal) = smooth.or_else(|| unit(cross(sub(b, a), sub(c, a)))) else { continue };
                let point = [f[0] + t * d[0], f[1] + t * d[1], f[2] + t * d[2]];
                let normal = world.apply_vector(normal);
                out.push(Hit {
                    occurrence: occurrence.to_owned(),
                    face: face.label.clone(),
                    reference: face.reference.clone(),
                    point: world.apply(point),
                    normal,
                    // The transform is rigid, so the distance along the unit direction carries over.
                    distance: t,
                    entering: dot(normal, dir) < 0.0,
                });
            }
        }
    }
}

/// Where the ray from `from` (m, world) along `direction` meets the part or assembly `path`,
/// nearest first. For an assembly, `only` limits the bodies to those occurrences (paths; a
/// sub-assembly's path takes in its parts). Coincident hits on shared triangle edges are kept once.
pub fn probe(
    path: &str,
    text: &str,
    files: &Files,
    from: V3,
    direction: V3,
    only: Option<&[String]>,
) -> Result<Vec<Hit>, String> {
    let dir = unit(direction).ok_or("the direction has no length")?;
    let mut hits = Vec::new();
    if !path.ends_with(".assembly") {
        let part = regenerate_part_in(path, text, None, &TessellationOptions::default())
            .map_err(|d| d.first().map(|d| d.message.clone()).unwrap_or_default())?;
        let display = part_display(&part.result, part.kernel.as_ref(), &TessellationOptions::default());
        hits_on(&display, &Transform::IDENTITY, "", from, dir, &mut hits);
    } else {
        let resolution = resolve_assembly_in(path, text, files, None)
            .map_err(|d| d.first().map(|d| d.message.clone()).unwrap_or_default())?;
        let resolved = &resolution.resolved;
        let wanted = |p: &str| only.is_none_or(|o| o.iter().any(|n| p == n || p.starts_with(&format!("{n}/"))));
        let mut displays: BTreeMap<(String, String), Option<PartDisplay>> = BTreeMap::new();
        let mut matched = false;
        for (id, node) in resolved.nodes.iter().enumerate() {
            if node.parent.is_none() || node.failed || node.represented_by_ancestor || node.definition.kind != DefinitionKind::Part {
                continue;
            }
            let occurrence = resolved.path(id as u32);
            if !wanted(&occurrence) {
                continue;
            }
            matched = true;
            let file = node.definition.file.0.clone();
            let key = crate::evaluate::content_key(&file, &node.configuration.digest(), files);
            // The world box, when an interference check already knows it, skips a body the ray misses.
            if let Some(bbox) = key.and_then(crate::evaluate::cached_bbox) {
                let world = bbox.transformed(&node.world);
                if !passes(from, dir, world.min, world.max) {
                    continue;
                }
            }
            let display = displays.entry((file.clone(), node.configuration.digest())).or_insert_with(|| {
                let document = resolution.documents.get(&file)?;
                let (result, kernel) = regenerate_configured(document, &node.configuration);
                Some(part_display(&result, kernel.as_ref(), &TessellationOptions::default()))
            });
            if let Some(display) = display {
                hits_on(display, &node.world, &occurrence, from, dir, &mut hits);
            }
        }
        if let (Some(names), false) = (only, matched) {
            return Err(format!("no part occurrence of the assembly is at {}", names.join(", ")));
        }
    }
    hits.sort_by(|a, b| a.distance.total_cmp(&b.distance));
    hits.dedup_by(|b, a| a.occurrence == b.occurrence && (a.distance - b.distance).abs() < 1e-9);
    Ok(hits)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_ray_meets_a_triangle_ahead_of_its_start_only() {
        let (a, b, c) = ([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        let (t, u, v) = meets([0.25, 0.25, 1.0], [0.0, 0.0, -1.0], a, b, c).unwrap();
        assert!((t - 1.0).abs() < 1e-12 && (u - 0.25).abs() < 1e-12 && (v - 0.25).abs() < 1e-12);
        assert!(meets([0.25, 0.25, 1.0], [0.0, 0.0, 1.0], a, b, c).is_none());
        assert!(meets([2.0, 2.0, 1.0], [0.0, 0.0, -1.0], a, b, c).is_none());
        assert!(passes([0.5, 0.5, 5.0], [0.0, 0.0, -1.0], [0.0; 3], [1.0; 3]));
        assert!(!passes([1.5, 0.5, 5.0], [0.0, 0.0, -1.0], [0.0; 3], [1.0; 3]));
    }
}
