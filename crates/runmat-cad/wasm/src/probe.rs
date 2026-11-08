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
use std::rc::Rc;

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

/// A ray: its start (m, world), its direction (any length) and, for an assembly, the
/// occurrences it is limited to (paths; a sub-assembly's path takes in its parts).
pub struct Ray<'a> {
    pub from: V3,
    pub direction: V3,
    pub only: Option<&'a [String]>,
}

/// Where each ray meets the part or assembly `path`, nearest first, one list per ray. The
/// assembly is resolved once for all the rays, and each body's display is regenerated once and
/// kept by content ([`displays`]), so a ray on an unchanged body costs only its triangle tests.
/// Coincident hits on shared triangle edges are kept once.
pub fn probe(path: &str, text: &str, files: &Files, rays: &[Ray]) -> Result<Vec<Vec<Hit>>, String> {
    let dirs = rays
        .iter()
        .map(|r| unit(r.direction).ok_or("a ray's direction has no length"))
        .collect::<Result<Vec<_>, _>>()?;
    let mut out: Vec<Vec<Hit>> = vec![Vec::new(); rays.len()];
    if !path.ends_with(".assembly") {
        let mut part_files = files.clone();
        part_files.insert(path.to_owned(), text.to_owned());
        let key = crate::evaluate::content_key(path, "", &part_files);
        let display = displays::get_or(key, || {
            let part = regenerate_part_in(path, text, None, &TessellationOptions::default())
                .map_err(|d| d.first().map(|d| d.message.clone()).unwrap_or_default())?;
            Ok::<_, String>(part_display(&part.result, part.kernel.as_ref(), &TessellationOptions::default()))
        })?;
        for ((ray, dir), hits) in rays.iter().zip(&dirs).zip(out.iter_mut()) {
            hits_on(&display, &Transform::IDENTITY, "", ray.from, *dir, hits);
        }
    } else {
        let resolution = resolve_assembly_in(path, text, files, None)
            .map_err(|d| d.first().map(|d| d.message.clone()).unwrap_or_default())?;
        let resolved = &resolution.resolved;
        let wanted = |only: Option<&[String]>, p: &str| {
            only.is_none_or(|o| o.iter().any(|n| p == n || p.starts_with(&format!("{n}/"))))
        };
        let mut matched = vec![false; rays.len()];
        for (id, node) in resolved.nodes.iter().enumerate() {
            if node.parent.is_none() || node.failed || node.represented_by_ancestor || node.definition.kind != DefinitionKind::Part {
                continue;
            }
            let occurrence = resolved.path(id as u32);
            let file = node.definition.file.0.clone();
            let key = crate::evaluate::content_key(&file, &node.configuration.digest(), files);
            // The world box, when an interference check already knows it, skips a body the ray misses.
            let world_box = key.and_then(crate::evaluate::cached_bbox).map(|b| b.transformed(&node.world));
            let mut display: Option<Rc<PartDisplay>> = None;
            for (i, (ray, dir)) in rays.iter().zip(&dirs).enumerate() {
                if !wanted(ray.only, &occurrence) {
                    continue;
                }
                matched[i] = true;
                if world_box.as_ref().is_some_and(|w| !passes(ray.from, *dir, w.min, w.max)) {
                    continue;
                }
                if display.is_none() {
                    let Some(document) = resolution.documents.get(&file) else { break };
                    display = Some(displays::get_or(key, || {
                        let (result, kernel) = regenerate_configured(document, &node.configuration);
                        Ok::<_, String>(part_display(&result, kernel.as_ref(), &TessellationOptions::default()))
                    })?);
                }
                if let Some(display) = &display {
                    hits_on(display, &node.world, &occurrence, ray.from, *dir, &mut out[i]);
                }
            }
        }
        for (ray, found) in rays.iter().zip(&matched) {
            if let (Some(names), false) = (ray.only, found) {
                return Err(format!("no part occurrence of the assembly is at {}", names.join(", ")));
            }
        }
    }
    for hits in &mut out {
        hits.sort_by(|a, b| a.distance.total_cmp(&b.distance));
        hits.dedup_by(|b, a| a.occurrence == b.occurrence && (a.distance - b.distance).abs() < 1e-9);
    }
    Ok(out)
}

/// The probe's displays (the default tessellation, finer than the assembly view's), kept by
/// content ([`crate::evaluate::content_key`]: the part's text, the files it reads and its
/// configuration), so the rays of one seating, cast over several calls, regenerate a curved
/// body once. A display is several megabytes, so only the last few are kept.
mod displays {
    use super::PartDisplay;
    use std::cell::RefCell;
    use std::rc::Rc;

    /// Displays kept: the bodies one seating rays (a shell, perhaps its mate).
    const KEPT: usize = 4;

    thread_local! {
        static KEPT_DISPLAYS: RefCell<Vec<(u64, Rc<PartDisplay>)>> = const { RefCell::new(Vec::new()) };
    }

    /// The display with content `key`, made by `make` when it isn't kept (or has no key).
    pub fn get_or<E>(key: Option<u64>, make: impl FnOnce() -> Result<PartDisplay, E>) -> Result<Rc<PartDisplay>, E> {
        let Some(key) = key else { return make().map(Rc::new) };
        if let Some(found) = KEPT_DISPLAYS.with(|k| {
            let mut k = k.borrow_mut();
            let at = k.iter().position(|(x, _)| *x == key)?;
            let entry = k.remove(at);
            let display = entry.1.clone();
            k.push(entry);
            Some(display)
        }) {
            return Ok(found);
        }
        let display = Rc::new(make()?);
        KEPT_DISPLAYS.with(|k| {
            let mut k = k.borrow_mut();
            if k.len() >= KEPT {
                k.remove(0);
            }
            k.push((key, display.clone()));
        });
        Ok(display)
    }

    #[cfg(test)]
    pub fn len() -> usize {
        KEPT_DISPLAYS.with(|k| k.borrow().len())
    }
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

    /// Several rays in one call each get their own hits, and a second call on the same text
    /// reuses the kept display instead of regenerating the body.
    #[test]
    fn rays_in_one_call_share_one_display() {
        let block = "runmat 0.7\nstd 1\n\npart block {\n    sketch s on origin.xy {\n        l1 = line(start = [0 mm, 0 mm], end = [20 mm, 0 mm])\n        l2 = line(start = l1.end, end = [20 mm, 10 mm])\n        l3 = line(start = l2.end, end = [0 mm, 10 mm])\n        l4 = line(start = l3.end, end = l1.start)\n    }\n    feature box = extrude(profile = sketch.s, distance = 5 mm)\n}\n";
        let rays = [
            Ray { from: [0.005, 0.005, -0.01], direction: [0.0, 0.0, 1.0], only: None },
            Ray { from: [0.015, 0.005, 0.02], direction: [0.0, 0.0, -1.0], only: None },
            Ray { from: [0.05, 0.005, -0.01], direction: [0.0, 0.0, 1.0], only: None },
        ];
        let first = probe("/p/block.part", block, &Files::new(), &rays).unwrap();
        assert_eq!(first.len(), 3);
        assert!(first[0][0].entering && (first[0][0].point[2]).abs() < 1e-9, "{:?}", first[0]);
        assert!(first[1][0].entering && (first[1][0].point[2] - 0.005).abs() < 1e-9, "{:?}", first[1]);
        assert!(first[2].is_empty(), "{:?}", first[2]);
        let kept = displays::len();
        let again = probe("/p/block.part", block, &Files::new(), &rays[..1]).unwrap();
        assert_eq!(again[0], first[0]);
        assert_eq!(displays::len(), kept, "the second call reuses the display");
    }
}
