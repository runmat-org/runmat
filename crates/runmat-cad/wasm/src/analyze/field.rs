//! Surface fields: a value at every vertex of a face's mesh, measured on the exact surface by
//! the kernel ([`AnalysisRequest::SurfaceField`]), which thickness, curvature and zebra color
//! the model by.
//!
//! The display tessellation is coarse where the surface is flat, so each face's triangles are
//! split (every edge at its midpoint, the same number of times over the whole face) until they
//! are short enough for the quantity: `target_edge` for thickness and curvature, and a fraction
//! of a stripe's turn of the normal for zebra. Each vertex is then projected onto the exact
//! surface; the field carries the projected positions, exact normals and values, and the Desktop
//! draws those faces from it while the result is open.

use runmat_cad::display::{BodyDisplay, FaceDisplay, PartDisplay};
use runmat_cad::evaluate::{
    AnalysisRequest, AnalysisResult, FacePoints, FieldPoint, FieldQuantity,
};
use serde::Serialize;
use std::collections::HashMap;

/// The most vertices one field carries. The field's faces are redrawn from it, eight numbers
/// a vertex (position, normal, value), and the scene the Desktop builds stays under its own
/// triangle budget of about a quarter million; 60 000 vertices keep the field near that scale
/// and its message near 4 MB.
pub(super) const FIELD_VERTEX_BUDGET: usize = 60_000;

/// The most times one face's triangles split (each split makes four triangles of one).
const MAX_LEVEL: u32 = 5;

/// A value range drawn as a color ramp, low to high, linear RGBA stops evenly spaced.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct SurfaceField {
    /// `spectrum` (a ramp with a range) or `stripes` (black and white bands).
    pub kind: &'static str,
    /// What the values are (`Thickness`, `Mean curvature`).
    pub label: String,
    /// The values' unit (`mm`, `1/mm`); empty for stripes.
    pub unit: String,
    /// The values at the ramp's ends, in `unit`.
    pub range: [f64; 2],
    /// The ramp's stops, low end first.
    pub ramp: Vec<[f32; 4]>,
    pub faces: Vec<FieldFace>,
}

/// One face drawn from the field: its mesh on the exact surface and a value at each vertex.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct FieldFace {
    /// The face's canonical reference, as the rows name it.
    pub item: String,
    /// x, y, z per vertex (metres).
    pub positions: Vec<f32>,
    /// The outward unit normal per vertex.
    pub normals: Vec<f32>,
    pub triangles: Vec<u32>,
    /// The value per vertex in the field's unit; `None` where it could not be measured (the
    /// face's own color shows there).
    pub values: Vec<Option<f32>>,
}

/// A face's mesh after splitting, with its body and face indices.
pub(super) struct FaceMesh<'a> {
    pub body: &'a BodyDisplay,
    pub face: &'a FaceDisplay,
    pub positions: Vec<[f64; 3]>,
    pub normals: Vec<[f64; 3]>,
    pub triangles: Vec<[u32; 3]>,
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn length(a: [f64; 3]) -> f64 {
    (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt()
}

fn mid(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        (a[0] + b[0]) / 2.0,
        (a[1] + b[1]) / 2.0,
        (a[2] + b[2]) / 2.0,
    ]
}

fn unit(a: [f64; 3]) -> [f64; 3] {
    let l = length(a);
    if l > 0.0 {
        [a[0] / l, a[1] / l, a[2] / l]
    } else {
        a
    }
}

/// The face's display mesh in f64.
fn display_mesh<'a>(body: &'a BodyDisplay, face: &'a FaceDisplay) -> FaceMesh<'a> {
    let positions: Vec<[f64; 3]> = face
        .positions
        .chunks_exact(3)
        .map(|p| [p[0] as f64, p[1] as f64, p[2] as f64])
        .collect();
    let normals = if face.normals.len() == face.positions.len() {
        face.normals
            .chunks_exact(3)
            .map(|n| [n[0] as f64, n[1] as f64, n[2] as f64])
            .collect()
    } else {
        Vec::new()
    };
    FaceMesh {
        body,
        face,
        positions,
        normals,
        triangles: face
            .triangles
            .chunks_exact(3)
            .map(|t| [t[0], t[1], t[2]])
            .collect(),
    }
}

/// Splits every triangle into four at its edges' midpoints, sharing midpoints along shared
/// edges (normals average and renormalize).
fn split(mesh: &mut FaceMesh<'_>) {
    let mut midpoints: HashMap<(u32, u32), u32> = HashMap::new();
    let has_normals = mesh.normals.len() == mesh.positions.len();
    let mut triangles = Vec::with_capacity(mesh.triangles.len() * 4);
    for t in std::mem::take(&mut mesh.triangles) {
        let mut m = [0u32; 3];
        for k in 0..3 {
            let (a, b) = (t[k], t[(k + 1) % 3]);
            let key = (a.min(b), a.max(b));
            m[k] = *midpoints.entry(key).or_insert_with(|| {
                mesh.positions
                    .push(mid(mesh.positions[a as usize], mesh.positions[b as usize]));
                if has_normals {
                    mesh.normals.push(unit(mid(
                        mesh.normals[a as usize],
                        mesh.normals[b as usize],
                    )));
                }
                (mesh.positions.len() - 1) as u32
            });
        }
        triangles.extend([
            [t[0], m[0], m[2]],
            [m[0], t[1], m[1]],
            [m[2], m[1], t[2]],
            [m[0], m[1], m[2]],
        ]);
    }
    mesh.triangles = triangles;
}

/// How many splits a face needs: until its longest edge is under `target_edge` and, when
/// `turn` is given, until no triangle's corner normals turn by more than `turn` radians.
fn level_for(mesh: &FaceMesh<'_>, target_edge: f64, turn: Option<f64>) -> u32 {
    let mut level = 0;
    let mut edge = 0.0f64;
    let mut angle = 0.0f64;
    let has_normals = mesh.normals.len() == mesh.positions.len();
    for t in &mesh.triangles {
        for k in 0..3 {
            let (a, b) = (t[k] as usize, t[(k + 1) % 3] as usize);
            edge = edge.max(length(sub(mesh.positions[a], mesh.positions[b])));
            if has_normals {
                let (na, nb) = (mesh.normals[a], mesh.normals[b]);
                let c = (na[0] * nb[0] + na[1] * nb[1] + na[2] * nb[2]).clamp(-1.0, 1.0);
                angle = angle.max(c.acos());
            }
        }
    }
    while level < MAX_LEVEL
        && (edge / f64::from(1u32 << level) > target_edge
            || turn.is_some_and(|turn| angle / f64::from(1u32 << level) > turn))
    {
        level += 1;
    }
    level
}

/// The faces of `bodies`' display, split for the field and kept under
/// [`FIELD_VERTEX_BUDGET`]: when the wanted splits would pass it, every face splits one time
/// fewer until they fit.
pub(super) fn field_meshes<'a>(
    bodies: &[&'a BodyDisplay],
    target_edge: f64,
    turn: Option<f64>,
) -> Vec<FaceMesh<'a>> {
    let base: Vec<FaceMesh<'a>> = bodies
        .iter()
        .flat_map(|b| b.faces.iter().map(move |f| display_mesh(b, f)))
        .collect();
    let mut levels: Vec<u32> = base
        .iter()
        .map(|m| level_for(m, target_edge, turn))
        .collect();
    // Each split multiplies a face's vertices by about four.
    let estimate = |levels: &[u32]| -> usize {
        base.iter()
            .zip(levels)
            .map(|(m, l)| m.positions.len() << (2 * l))
            .sum()
    };
    while estimate(&levels) > FIELD_VERTEX_BUDGET && levels.iter().any(|l| *l > 0) {
        for l in &mut levels {
            *l = l.saturating_sub(1);
        }
    }
    base.into_iter()
        .zip(levels)
        .map(|(mut m, l)| {
            for _ in 0..l {
                split(&mut m);
            }
            m
        })
        .collect()
}

/// Each vertex moved a little into the face, toward the middle of a triangle it is a corner of:
/// a ray from a point exactly on the face's boundary can run along the neighbouring face, and
/// a projection from there can land on the face's seam.
fn nudged(mesh: &FaceMesh<'_>, by: f64) -> Vec<[f64; 3]> {
    let mut toward: Vec<Option<[f64; 3]>> = vec![None; mesh.positions.len()];
    for t in &mesh.triangles {
        let c = [0, 1, 2].map(|k| {
            (mesh.positions[t[0] as usize][k]
                + mesh.positions[t[1] as usize][k]
                + mesh.positions[t[2] as usize][k])
                / 3.0
        });
        for &v in t {
            toward[v as usize].get_or_insert(c);
        }
    }
    mesh.positions
        .iter()
        .zip(toward)
        .map(|(p, c)| match c {
            Some(c) => {
                let d = sub(c, *p);
                let l = length(d);
                if l <= 0.0 {
                    *p
                } else {
                    let s = by.min(l / 4.0) / l;
                    [p[0] + d[0] * s, p[1] + d[1] * s, p[2] + d[2] * s]
                }
            }
            None => *p,
        })
        .collect()
}

/// The kernel's points for each mesh, per body (one request per body).
pub(super) fn measure(
    part: &crate::runtime::RegeneratedPart,
    meshes: &[FaceMesh<'_>],
    quantity: FieldQuantity,
) -> Result<Vec<Vec<FieldPoint>>, String> {
    let size = part
        .display
        .bounds
        .as_ref()
        .map(|b| length(sub(b.max.map(f64::from), b.min.map(f64::from))))
        .unwrap_or(1.0);
    let by = size * 1e-4;
    let mut out: Vec<Vec<FieldPoint>> = vec![Vec::new(); meshes.len()];
    let mut bodies: Vec<u32> = meshes.iter().map(|m| m.body.index).collect();
    bodies.dedup();
    for body in bodies {
        let at: Vec<usize> = (0..meshes.len())
            .filter(|i| meshes[*i].body.index == body)
            .collect();
        let Some(state) = part.result.state.bodies.get(body as usize) else {
            continue;
        };
        let request = AnalysisRequest::SurfaceField {
            quantity,
            faces: at
                .iter()
                .map(|i| FacePoints {
                    face: meshes[*i].face.index,
                    points: nudged(&meshes[*i], by),
                })
                .collect(),
        };
        match part.kernel.analyze(&request, &[&state.handle]) {
            Ok(AnalysisResult::SurfaceField { faces }) => {
                for (i, points) in at.into_iter().zip(faces) {
                    out[i] = points;
                }
            }
            Ok(other) => return Err(format!("the kernel answered {other:?}")),
            Err(e) => return Err(format!("the kernel could not measure the faces: {e:?}")),
        }
    }
    Ok(out)
}

/// A face drawn from its mesh, the kernel's points and a value per point.
pub(super) fn field_face(
    mesh: &FaceMesh<'_>,
    points: &[FieldPoint],
    value: impl Fn(&FieldPoint) -> Option<f64>,
) -> Option<FieldFace> {
    let item = mesh.face.reference.clone()?;
    let mut positions = Vec::with_capacity(mesh.positions.len() * 3);
    let mut normals = Vec::with_capacity(mesh.positions.len() * 3);
    let mut values = Vec::with_capacity(mesh.positions.len());
    for (i, p) in mesh.positions.iter().enumerate() {
        let point = points.get(i);
        let at = point.and_then(|q| q.point).unwrap_or(*p);
        let n = point
            .and_then(|q| q.normal)
            .or_else(|| mesh.normals.get(i).copied())
            .unwrap_or([0.0; 3]);
        positions.extend(at.map(|c| c as f32));
        normals.extend(n.map(|c| c as f32));
        values.push(point.and_then(&value).map(|v| v as f32));
    }
    Some(FieldFace {
        item,
        positions,
        normals,
        triangles: mesh.triangles.iter().flatten().copied().collect(),
        values,
    })
}

/// A ramp for values where low is bad (thin walls): red, amber, green, teal, blue.
pub(super) const RAMP_LOW_RED: [[f32; 4]; 5] = [
    [0.86, 0.20, 0.18, 1.0],
    [0.95, 0.66, 0.16, 1.0],
    [0.30, 0.72, 0.36, 1.0],
    [0.18, 0.62, 0.70, 1.0],
    [0.22, 0.40, 0.86, 1.0],
];

/// A ramp for signed values: blue (negative), green-grey (zero), red (positive).
pub(super) const RAMP_SIGNED: [[f32; 4]; 5] = [
    [0.20, 0.36, 0.86, 1.0],
    [0.36, 0.70, 0.90, 1.0],
    [0.72, 0.78, 0.72, 1.0],
    [0.96, 0.66, 0.30, 1.0],
    [0.86, 0.20, 0.18, 1.0],
];

/// Zebra's two bands.
pub(super) const RAMP_STRIPES: [[f32; 4]; 2] = [[0.06, 0.06, 0.07, 1.0], [0.97, 0.97, 0.97, 1.0]];

/// The least and greatest finite values of the faces (in the field's unit).
pub(super) fn value_range(faces: &[FieldFace]) -> Option<[f64; 2]> {
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for v in faces.iter().flat_map(|f| f.values.iter().flatten()) {
        lo = lo.min(f64::from(*v));
        hi = hi.max(f64::from(*v));
    }
    (lo <= hi).then_some([lo, hi])
}

/// A model's size: the diagonal of its bounds (metres).
pub(super) fn model_size(display: &PartDisplay) -> f64 {
    display
        .bounds
        .as_ref()
        .map(|b| length(sub(b.max.map(f64::from), b.min.map(f64::from))))
        .unwrap_or(0.1)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn square() -> (BodyDisplay, FaceDisplay) {
        let face = FaceDisplay {
            index: 0,
            reference: Some("feature.body.face.top".into()),
            positions: vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
            triangles: vec![0, 1, 2, 0, 2, 3],
            normals: vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
            ..Default::default()
        };
        (BodyDisplay::default(), face)
    }

    #[test]
    fn a_split_shares_midpoints_and_quarters_the_triangles() {
        let (body, face) = square();
        let mut mesh = display_mesh(&body, &face);
        split(&mut mesh);
        assert_eq!(mesh.triangles.len(), 8);
        // 4 corners + 5 midpoints (the diagonal's is shared).
        assert_eq!(mesh.positions.len(), 9);
        assert_eq!(mesh.normals.len(), 9);
    }

    #[test]
    fn faces_split_until_edges_are_short_enough() {
        let (body, face) = square();
        let mesh = display_mesh(&body, &face);
        // The diagonal is √2 long: three halvings bring it under 0.2.
        assert_eq!(level_for(&mesh, 0.2, None), 3);
        assert_eq!(level_for(&mesh, 2.0, None), 0);
    }

    #[test]
    fn the_budget_caps_the_splits() {
        let (body, mut face) = square();
        face.reference = Some("a".into());
        let mut b = body.clone();
        b.faces = vec![face];
        let meshes = field_meshes(&[&b], 1e-9, None);
        assert!(meshes[0].positions.len() <= FIELD_VERTEX_BUDGET);
        assert!(meshes[0].triangles.len() >= 2 << (2 * (MAX_LEVEL - 1)));
    }

    #[test]
    fn nudged_points_move_into_the_face() {
        let (body, face) = square();
        let mesh = display_mesh(&body, &face);
        let p = nudged(&mesh, 0.01);
        assert!(p[0][0] > 0.0 && p[0][1] > 0.0 && p[0][2] == 0.0);
        assert!(length(sub(p[0], mesh.positions[0])) <= 0.01 + 1e-12);
    }
}
