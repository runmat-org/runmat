//! `assembly.contacts`: contact regions between components' faces (`contact <name> = { a, b,
//! bonded }`), found by scanning every pair of components, and Delete on a `contact:<name>` row.
//!
//! Two faces are in contact when they lie within the gap tolerance of each other and overlap:
//!
//! - two planar faces facing each other (opposite normals), the gap between their planes within
//!   the tolerance, and part of one face lying on the other;
//! - two cylindrical faces on the same axis (within the tolerance) with radii within the
//!   tolerance of each other, overlapping along the axis (a pin in its hole).
//!
//! Each face is published from its part, so the region follows the part's edits. A pair of faces
//! already named by a region is left as it is. Run on `contact:<name>` rows, the command changes
//! whether those regions are bonded. Reference and suppressed components are left out.

use super::anchors::{Anchor, Publisher};
use super::frames::{
    cross, dot, face_frame, indices, points, sub, to_world, unit, Frame, FrameKind, V3,
};
use super::mate::field;
use super::{model, Compiled};
use runmat_cad::assembly::model::{AssemblyDocument, ContactDecl, ContactFace, RelPath};
use runmat_cad::refs::TopoKind;
use runmat_cad::syntax::edit::EditFailure;
use serde_json::{Map, Value};

type Values = Map<String, Value>;

/// A component's face in assembly coordinates.
struct PlacedFace {
    frame: Frame,
    /// The face's triangles in world coordinates.
    triangles: Vec<[V3; 3]>,
    reference: String,
}

/// The faces of top-level component `name` that can be published (they carry a reference), in
/// world coordinates, with its part file.
fn placed_faces(compiled: &Compiled, name: &str) -> Option<(String, Vec<PlacedFace>)> {
    let (prototype, t) = compiled.placed_named(name)?;
    let file = compiled.prototype_file(prototype)?;
    let bodies = compiled.display()["prototypes"][prototype]["part"]["bodies"].as_array()?;
    let world = |p: V3| to_world(point_frame(p), &t).origin;
    let mut out = Vec::new();
    for face in bodies
        .iter()
        .flat_map(|b| b["faces"].as_array().into_iter().flatten())
    {
        let (Some(frame), Some(reference)) = (face_frame(face), face["reference"].as_str()) else {
            continue;
        };
        let p = points(&face["positions"]);
        let triangles = indices(&face["triangles"])
            .chunks_exact(3)
            .filter_map(|c| {
                Some([
                    world(*p.get(c[0])?),
                    world(*p.get(c[1])?),
                    world(*p.get(c[2])?),
                ])
            })
            .collect();
        out.push(PlacedFace {
            frame: to_world(frame, &t),
            triangles,
            reference: reference.to_owned(),
        });
    }
    Some((file, out))
}

fn point_frame(origin: V3) -> Frame {
    Frame {
        kind: FrameKind::Point,
        origin,
        z: [0.0, 0.0, 1.0],
        x: [1.0, 0.0, 0.0],
        radius: None,
    }
}

/// `p` inside triangle `t`, all in one plane's (u, v) coordinates.
fn inside(p: [f64; 2], t: [[f64; 2]; 3]) -> bool {
    let side =
        |a: [f64; 2], b: [f64; 2]| (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]);
    let (d1, d2, d3) = (side(t[0], t[1]), side(t[1], t[2]), side(t[2], t[0]));
    let eps = 1e-12;
    !((d1 < -eps || d2 < -eps || d3 < -eps) && (d1 > eps || d2 > eps || d3 > eps))
}

/// Whether part of one planar face lies on the other: a triangle's centroid of either face
/// inside a triangle of the other, in the first face's plane.
fn planes_overlap(a: &PlacedFace, b: &PlacedFace) -> bool {
    let u = unit(a.frame.x);
    let v = unit(cross(a.frame.z, u));
    let flat = |p: V3| {
        let d = sub(p, a.frame.origin);
        [dot(d, u), dot(d, v)]
    };
    let flat_tris = |f: &PlacedFace| -> Vec<[[f64; 2]; 3]> {
        f.triangles.iter().map(|t| t.map(flat)).collect()
    };
    let (ta, tb) = (flat_tris(a), flat_tris(b));
    let centroid = |t: &[[f64; 2]; 3]| {
        [
            (t[0][0] + t[1][0] + t[2][0]) / 3.0,
            (t[0][1] + t[1][1] + t[2][1]) / 3.0,
        ]
    };
    ta.iter()
        .any(|t| tb.iter().any(|s| inside(centroid(t), *s)))
        || tb
            .iter()
            .any(|t| ta.iter().any(|s| inside(centroid(t), *s)))
}

/// The interval a face covers along an axis through `origin` along `z`.
fn along(f: &PlacedFace, origin: V3, z: V3) -> (f64, f64) {
    f.triangles
        .iter()
        .flatten()
        .map(|p| dot(sub(*p, origin), z))
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), s| {
            (lo.min(s), hi.max(s))
        })
}

/// Whether faces `a` and `b` touch within `tol` (m).
fn touching(a: &PlacedFace, b: &PlacedFace, tol: f64) -> bool {
    match (a.frame.kind, b.frame.kind) {
        (FrameKind::Plane, FrameKind::Plane) => {
            dot(unit(a.frame.z), unit(b.frame.z)) < -0.999
                && dot(sub(b.frame.origin, a.frame.origin), unit(a.frame.z)).abs() <= tol
                && planes_overlap(a, b)
        }
        (FrameKind::Axis, FrameKind::Axis) => {
            let (Some(ra), Some(rb)) = (a.frame.radius, b.frame.radius) else {
                return false;
            };
            let z = unit(a.frame.z);
            if dot(z, unit(b.frame.z)).abs() < 0.999 || (ra - rb).abs() > tol {
                return false;
            }
            let d = sub(b.frame.origin, a.frame.origin);
            let off = sub(d, z.map(|c| c * dot(d, z)));
            if dot(off, off).sqrt() > tol {
                return false;
            }
            let (a0, a1) = along(a, a.frame.origin, z);
            let (b0, b1) = along(b, a.frame.origin, z);
            a1.min(b1) - a0.max(b0) > tol
        }
        _ => false,
    }
}

pub(super) fn selected(request: &super::EditRequest) -> impl Iterator<Item = &str> {
    request
        .selection
        .iter()
        .filter_map(|id| id.strip_prefix("contact:"))
}

fn bonded(values: &Values) -> bool {
    values
        .get("bonded")
        .and_then(Value::as_bool)
        .unwrap_or(true)
}

/// `assembly.contacts` on `contact:<name>` rows: sets whether they are bonded.
pub(super) fn set_bonded(
    document: &mut AssemblyDocument,
    request: &super::EditRequest,
) -> Result<String, EditFailure> {
    let want = bonded(&request.values);
    let mut names = Vec::new();
    for name in selected(request) {
        let c = document
            .contacts
            .iter_mut()
            .find(|c| c.name == name)
            .ok_or_else(|| model(format!("the assembly has no contact region `{name}`")))?;
        c.bonded = want;
        names.push(name.to_owned());
    }
    Ok(format!(
        "Made {} {}",
        names.join(", "),
        if want { "bonded" } else { "free to slide" }
    ))
}

/// `assembly.contacts`: names every touching pair of faces not named yet.
pub(super) fn find(
    document: &mut AssemblyDocument,
    compiled: &Compiled,
    publisher: &mut Publisher,
    values: &Values,
) -> Result<String, EditFailure> {
    let tol = match values.get("tolerance") {
        None | Some(Value::Null) => 0.05e-3,
        Some(Value::String(s)) if s.trim().is_empty() => 0.05e-3,
        _ => {
            let v = field(values, "tolerance", "mm")?;
            runmat_cad::units::length_mm(&v)
                .filter(|t| *t >= 0.0)
                .ok_or_else(|| model("the gap tolerance must be a length of zero or more"))?
                / 1e3
        }
    };
    let bonded = bonded(values);
    let names: Vec<String> = document
        .occurrences
        .iter()
        .filter(|o| !o.suppressed && !o.reference)
        .map(|o| o.name.clone())
        .collect();
    let faces: Vec<(String, String, Vec<PlacedFace>)> = names
        .iter()
        .filter_map(|n| placed_faces(compiled, n).map(|(file, f)| (n.clone(), file, f)))
        .collect();
    let mut added = Vec::new();
    for (i, (na, fa, a)) in faces.iter().enumerate() {
        for (nb, fb, b) in faces.iter().skip(i + 1) {
            for x in a {
                for y in b.iter().filter(|y| touching(x, y, tol)) {
                    let side =
                        |occ: &str, file: &str, face: &PlacedFace, publisher: &mut Publisher| {
                            let anchor = Anchor::Entity {
                                file: file.to_owned(),
                                kind: TopoKind::Face,
                                reference: face.reference.clone(),
                            };
                            let name = publisher.name_of(&anchor)?.ok_or_else(|| {
                                model(format!("a face of `{occ}` can't be published"))
                            })?;
                            Ok::<_, EditFailure>(ContactFace {
                                occurrence: RelPath::new(&[occ]),
                                face: name,
                            })
                        };
                    let ca = side(na, fa, x, publisher)?;
                    let cb = side(nb, fb, y, publisher)?;
                    let known = document
                        .contacts
                        .iter()
                        .any(|c| (c.a == ca && c.b == cb) || (c.a == cb && c.b == ca));
                    if known {
                        continue;
                    }
                    let name = (1..)
                        .map(|n| format!("contact{n}"))
                        .find(|n| !document.contacts.iter().any(|c| &c.name == n))
                        .expect("an unused name");
                    added.push(format!("{na}–{nb}"));
                    document.contacts.push(ContactDecl {
                        name,
                        a: ca,
                        b: cb,
                        bonded,
                    });
                }
            }
        }
    }
    if added.is_empty() {
        return Err(model(format!(
            "no faces of different components touch within {} mm that aren't named already",
            (tol * 1e6).round() / 1e3
        )));
    }
    Ok(format!(
        "Found {} contact region{} ({})",
        added.len(),
        if added.len() == 1 { "" } else { "s" },
        added.join(", ")
    ))
}

/// `assembly.delete` on selected contact rows.
pub(super) fn delete(
    document: &mut AssemblyDocument,
    request: &super::EditRequest,
) -> Result<String, EditFailure> {
    let mut names = Vec::new();
    for name in selected(request) {
        let at = document
            .contacts
            .iter()
            .position(|c| c.name == name)
            .ok_or_else(|| model(format!("the assembly has no contact region `{name}`")))?;
        document.contacts.remove(at);
        names.push(name.to_owned());
    }
    Ok(format!("Deleted contact region {}", names.join(", ")))
}
