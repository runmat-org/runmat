//! A planar face of a part as DXF (millimetres): the face's outline and its holes, seen along the
//! face's normal, as closed polylines on the OUTLINE layer. The loops are the boundary of the
//! kernel's tessellation of the face (edges that only one triangle uses), so they lie on the
//! model's edges; a curved edge is written as its tessellation's straight segments.

use crate::formats::sheet::{standard_linetypes, Layer, Shape, Sheet};
use crate::ToolError;
use std::collections::BTreeMap;

fn fail(message: String) -> ToolError {
    ToolError::new("invalid_arguments", message)
}

/// The DXF text of the face `pick` (its reference text, as the canvas picks it) of the part.
pub fn face_dxf(path: &str, text: &str, pick: &str, configuration: Option<&str>) -> Result<String, ToolError> {
    let part = runmat_cad_wasm::runtime::regenerate_part_in(path, text, configuration, &Default::default()).map_err(|d| {
        ToolError::new("not_compiled", format!("{path} doesn't compile: {}", d.first().map(|d| d.message.as_str()).unwrap_or("")))
    })?;
    let face = part
        .display
        .bodies
        .iter()
        .flat_map(|b| &b.faces)
        .find(|f| f.reference.as_deref() == Some(pick) || f.names.iter().any(|n| n == pick))
        .ok_or_else(|| fail(format!("the part has no face `{pick}`")))?;
    let points: Vec<[f64; 3]> = face.positions.chunks_exact(3).map(|c| [c[0] as f64, c[1] as f64, c[2] as f64]).collect();
    let loops = outline_loops(&points, &face.triangles);
    let normal = newell(&points, &face.triangles);
    let flat = points.iter().all(|p| ((sub(*p, points[0])).iter().zip(normal).map(|(a, b)| a * b).sum::<f64>()).abs() < 1e-6);
    if loops.is_empty() || !flat {
        return Err(fail(format!("DXF of a face takes a planar face; `{pick}` is curved")));
    }
    // In-plane axes: x along the world axis least aligned with the normal, y = n × x.
    let seed = if normal[0].abs() < 0.9 { [1.0, 0.0, 0.0] } else { [0.0, 1.0, 0.0] };
    let x = unit(sub(seed, scale(normal, dot(seed, normal))));
    let y = cross(normal, x);
    let to_mm = |p: [f64; 3]| [dot(p, x) * 1e3, dot(p, y) * 1e3];
    let mut sheet = Sheet {
        layers: vec![Layer { name: "OUTLINE".into(), color: 7, linetype: "CONTINUOUS".into() }],
        linetypes: standard_linetypes(),
        ..Sheet::default()
    };
    let (mut lo, mut hi) = ([f64::INFINITY; 2], [f64::NEG_INFINITY; 2]);
    for ring in &loops {
        let pts: Vec<[f64; 2]> = ring.iter().map(|&i| to_mm(points[i])).collect();
        for p in &pts {
            lo = [lo[0].min(p[0]), lo[1].min(p[1])];
            hi = [hi[0].max(p[0]), hi[1].max(p[1])];
        }
        sheet.add("OUTLINE", Shape::Polyline { points: pts, closed: true });
    }
    // The outline's lower-left corner at the origin.
    for e in &mut sheet.entities {
        if let Shape::Polyline { points, .. } = &mut e.shape {
            for p in points.iter_mut() {
                *p = [p[0] - lo[0], p[1] - lo[1]];
            }
        }
    }
    sheet.paper = [hi[0] - lo[0], hi[1] - lo[1]];
    Ok(super::dxf::write(&sheet))
}

/// The closed boundary loops of a triangulated face, vertices welded by position (1e-9 m).
fn outline_loops(points: &[[f64; 3]], triangles: &[u32]) -> Vec<Vec<usize>> {
    let mut weld: BTreeMap<[i64; 3], usize> = BTreeMap::new();
    let id: Vec<usize> = points.iter().enumerate().map(|(i, p)| *weld.entry(p.map(|v| (v * 1e9).round() as i64)).or_insert(i)).collect();
    let mut count: BTreeMap<(usize, usize), (usize, usize)> = BTreeMap::new();
    for t in triangles.chunks_exact(3) {
        let v = [id[t[0] as usize], id[t[1] as usize], id[t[2] as usize]];
        for k in 0..3 {
            let (a, b) = (v[k], v[(k + 1) % 3]);
            if a == b {
                continue;
            }
            let key = (a.min(b), a.max(b));
            let e = count.entry(key).or_insert((0, 0));
            e.0 += 1;
            e.1 = a; // the directed start of the last use
        }
    }
    let mut next: BTreeMap<usize, usize> = BTreeMap::new();
    for (&(a, b), &(n, start)) in &count {
        if n == 1 {
            let end = if start == a { b } else { a };
            next.insert(start, end);
        }
    }
    let mut loops = Vec::new();
    while let Some((&first, _)) = next.iter().next() {
        let mut ring = vec![first];
        let mut at = next.remove(&first).unwrap_or(first);
        while at != first {
            ring.push(at);
            match next.remove(&at) {
                Some(n) => at = n,
                None => break,
            }
        }
        if ring.len() >= 3 {
            loops.push(ring);
        }
    }
    loops
}

fn newell(points: &[[f64; 3]], triangles: &[u32]) -> [f64; 3] {
    let mut n = [0.0; 3];
    for t in triangles.chunks_exact(3) {
        let (a, b, c) = (points[t[0] as usize], points[t[1] as usize], points[t[2] as usize]);
        let m = cross(sub(b, a), sub(c, a));
        n = [n[0] + m[0], n[1] + m[1], n[2] + m[2]];
    }
    unit(n)
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn scale(a: [f64; 3], s: f64) -> [f64; 3] {
    a.map(|v| v * s)
}
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
fn unit(a: [f64; 3]) -> [f64; 3] {
    let l = dot(a, a).sqrt();
    if l > 0.0 { a.map(|v| v / l) } else { [0.0, 0.0, 1.0] }
}
