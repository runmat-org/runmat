//! Measure's picks as geometry in world coordinates (metres).
//!
//! A part's picks are its display's reference texts (`feature.body.face.end`), topology picks
//! (`face:<body>:<face>`, `edge:…`, `vertex:…`) or `body:<index>`. An assembly's are
//! `occurrence:<uuid>` for a component (every part body at or under it) and
//! `occurrence:<uuid>/face:<body>:<face>` (or `edge:`, `vertex:`) for an entity of one, indexed
//! into the display of the component's part.

use super::distance::Primitives;
use crate::edit::assembly::frames::{add, cross, dot, edge_frame, norm, scale, sub, unit, V3};
use runmat_cad::display::{BodyDisplay, EdgeDisplay, FaceDisplay, PartDisplay};
use runmat_cad::refs::history::SurfaceKind;
use runmat_cad::regen::kernel::Transform;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Vertex,
    Edge,
    Face,
    Body,
}

/// A straight line, a plane or an axis the item has, for angles and normal distances.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Direction {
    /// A straight edge: a point on it and its unit direction.
    Line(V3, V3),
    /// A planar face: a point on it and its unit normal.
    Plane(V3, V3),
    /// A cylindrical face: a point on its axis and the axis's unit direction. (A circular edge
    /// reads as the plane it lies in; its axis is in [`Circle`].)
    Axis(V3, V3),
}

/// A circle the item lies on: a circular edge's, or a cylindrical face's cross-section.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Circle {
    pub centre: V3,
    pub axis: V3,
    pub radius: f64,
}

#[derive(Clone, Debug)]
pub struct Item {
    pub id: String,
    pub label: String,
    pub kind: Kind,
    pub prims: Primitives,
    pub direction: Option<Direction>,
    pub circle: Option<Circle>,
    /// An edge's length, a face's area (m, m²).
    pub length: Option<f64>,
    pub area: Option<f64>,
    /// A point to hang a one-item dimension's label on.
    pub anchor: V3,
}

/// One part display placed in the model: a part file's own (identity), or a component's.
pub struct Placed<'a> {
    pub display: &'a PartDisplay,
    pub world: Transform,
    /// The component's path, which labels prefix (empty for a part file).
    pub occurrence: String,
}

fn f(p: [f32; 3]) -> V3 {
    p.map(f64::from)
}

fn chunks(v: &[f32]) -> Vec<V3> {
    v.chunks_exact(3)
        .map(|c| [c[0] as f64, c[1] as f64, c[2] as f64])
        .collect()
}

fn prefixed(at: &Placed<'_>, label: String) -> String {
    if at.occurrence.is_empty() {
        label
    } else {
        format!("{} · {label}", at.occurrence)
    }
}

fn face_label(face: &FaceDisplay) -> String {
    crate::face_labels::label(0, face)
}

fn triangles(at: &Placed<'_>, face: &FaceDisplay) -> Vec<[V3; 3]> {
    let p = chunks(&face.positions);
    face.triangles
        .chunks_exact(3)
        .filter_map(|t| {
            Some(
                [
                    *p.get(t[0] as usize)?,
                    *p.get(t[1] as usize)?,
                    *p.get(t[2] as usize)?,
                ]
                .map(|q| at.world.apply(q)),
            )
        })
        .collect()
}

pub fn face_item(at: &Placed<'_>, id: &str, face: &FaceDisplay) -> Item {
    let tris = triangles(at, face);
    let (mut area, mut centroid, mut normal) = (0.0, [0.0; 3], [0.0; 3]);
    let mut flat = true;
    let mut first: Option<V3> = None;
    for t in &tris {
        let n = cross(sub(t[1], t[0]), sub(t[2], t[0]));
        let w = norm(n) / 2.0;
        if w <= 0.0 {
            continue;
        }
        let u = unit(n);
        match first {
            None => first = Some(u),
            Some(f) if dot(f, u) < 0.9998 => flat = false,
            _ => {}
        }
        area += w;
        centroid = add(centroid, scale(add(add(t[0], t[1]), t[2]), w / 3.0));
        normal = add(normal, scale(n, 0.5));
    }
    let centroid = if area > 0.0 {
        scale(centroid, 1.0 / area)
    } else {
        tris.first().map_or([0.0; 3], |t| t[0])
    };
    let (direction, circle) = match face.surface {
        Some(SurfaceKind::Plane { normal: n }) => (
            Some(Direction::Plane(centroid, unit(at.world.apply_vector(n)))),
            None,
        ),
        Some(SurfaceKind::Cylinder {
            axis_point,
            axis_direction,
            radius,
        }) => {
            let (p, d) = (
                at.world.apply(axis_point),
                unit(at.world.apply_vector(axis_direction)),
            );
            // The axis point level with the face's middle.
            let centre = add(p, scale(d, dot(sub(centroid, p), d)));
            (
                Some(Direction::Axis(centre, d)),
                Some(Circle {
                    centre,
                    axis: d,
                    radius,
                }),
            )
        }
        _ if flat && area > 0.0 => (Some(Direction::Plane(centroid, unit(normal))), None),
        _ => (None, None),
    };
    Item {
        id: id.into(),
        label: prefixed(at, face_label(face)),
        kind: Kind::Face,
        anchor: centroid,
        prims: Primitives {
            triangles: tris,
            ..Default::default()
        },
        direction,
        circle,
        length: None,
        area: Some(area),
    }
}

/// An edge's name for people: the two faces it runs between.
fn edge_label(body: &BodyDisplay, edge: &EdgeDisplay) -> String {
    let name = |i: u32| body.faces.iter().find(|f| f.index == i).map(face_label);
    match (name(edge.faces[0]), name(edge.faces[1])) {
        (Some(a), Some(b)) if a != b => format!("Edge of {a} and {b}"),
        (Some(a), _) | (_, Some(a)) => format!("Edge of {a}"),
        _ => format!("Edge {}", edge.index + 1),
    }
}

pub fn edge_item(at: &Placed<'_>, id: &str, body: &BodyDisplay, edge: &EdgeDisplay) -> Item {
    let pts: Vec<V3> = chunks(&edge.points)
        .into_iter()
        .map(|q| at.world.apply(q))
        .collect();
    let segments: Vec<[V3; 2]> = pts.windows(2).map(|w| [w[0], w[1]]).collect();
    let chord_length: f64 = segments.iter().map(|s| norm(sub(s[1], s[0]))).sum();
    let flat: Vec<f64> = pts.iter().flatten().copied().collect();
    let frame = edge_frame(&serde_json::json!({ "points": flat }));
    let (mut direction, mut circle, mut length) = (None, None, chord_length);
    if let Some(frame) = frame {
        match frame.radius {
            None => direction = Some(Direction::Line(frame.origin, frame.z)),
            Some(r) => {
                // A circle only when every point lies on it; a spline's fit is no circle.
                let on = pts.iter().all(|p| {
                    let d = sub(*p, frame.origin);
                    (norm(d) - r).abs() <= 1e-4 * r + 1e-9
                        && dot(d, frame.z).abs() <= 1e-4 * r + 1e-9
                });
                if on && pts.len() >= 3 {
                    circle = Some(Circle {
                        centre: frame.origin,
                        axis: frame.z,
                        radius: r,
                    });
                    // A circle lies in a plane: angles and normal distances read it as one.
                    direction = Some(Direction::Plane(frame.origin, frame.z));
                    // The arc's length, from the angle each chord turns through.
                    length = pts
                        .windows(2)
                        .map(|w| {
                            let (a, b) =
                                (unit(sub(w[0], frame.origin)), unit(sub(w[1], frame.origin)));
                            norm(cross(a, b)).atan2(dot(a, b)) * r
                        })
                        .sum();
                }
            }
        }
    }
    let anchor = pts.get(pts.len() / 2).copied().unwrap_or([0.0; 3]);
    Item {
        id: id.into(),
        label: prefixed(at, edge_label(body, edge)),
        kind: Kind::Edge,
        anchor,
        prims: Primitives {
            segments,
            ..Default::default()
        },
        direction,
        circle,
        length: Some(length),
        area: None,
    }
}

pub fn vertex_item(at: &Placed<'_>, id: &str, index: u32, point: [f32; 3]) -> Item {
    let p = at.world.apply(f(point));
    Item {
        id: id.into(),
        label: prefixed(at, format!("Vertex {}", index + 1)),
        kind: Kind::Vertex,
        anchor: p,
        prims: Primitives {
            points: vec![p],
            ..Default::default()
        },
        direction: None,
        circle: None,
        length: None,
        area: None,
    }
}

/// Every body of `at` as one item (a part body, or a component).
pub fn bodies_item(at: &[Placed<'_>], id: &str, label: String, only: Option<u32>) -> Option<Item> {
    let mut prims = Primitives::default();
    for placed in at {
        for body in placed
            .display
            .bodies
            .iter()
            .filter(|b| only.is_none_or(|i| b.index == i))
        {
            for face in &body.faces {
                prims.triangles.extend(triangles(placed, face));
            }
        }
    }
    let anchor = prims.triangles.first()?[0];
    Some(Item {
        id: id.into(),
        label,
        kind: Kind::Body,
        anchor,
        prims,
        direction: None,
        circle: None,
        length: None,
        area: None,
    })
}

/// `kind:<body>:<index>`.
fn topology(pick: &str) -> Option<(&str, u32, u32)> {
    let (kind, rest) = pick.split_once(':')?;
    let (body, index) = rest.split_once(':')?;
    Some((kind, body.parse().ok()?, index.parse().ok()?))
}

/// A pick on one placed part display: a reference text, a topology pick, or `body:<n>`.
pub fn on_part(at: &Placed<'_>, id: &str, pick: &str) -> Option<Item> {
    let body = |b: u32| at.display.bodies.iter().find(|x| x.index == b);
    if let Some((kind, b, i)) = topology(pick) {
        let body = body(b)?;
        return match kind {
            "face" => body
                .faces
                .iter()
                .find(|f| f.index == i)
                .map(|f| face_item(at, id, f)),
            "edge" => body
                .edges
                .iter()
                .find(|e| e.index == i)
                .map(|e| edge_item(at, id, body, e)),
            "vertex" => body
                .vertices
                .iter()
                .find(|v| v.index == i)
                .map(|v| vertex_item(at, id, v.index, v.point)),
            _ => None,
        };
    }
    if let Some(b) = pick
        .strip_prefix("body:")
        .and_then(|b| b.parse::<u32>().ok())
    {
        body(b)?;
        let label = prefixed(at, format!("Body {}", b + 1));
        return bodies_item(std::slice::from_ref(at), id, label, Some(b));
    }
    let is = |r: &Option<String>| r.as_deref() == Some(pick);
    for body in &at.display.bodies {
        if let Some(face) = body
            .faces
            .iter()
            .find(|f| is(&f.reference) || f.names.iter().any(|n| n == pick))
        {
            return Some(face_item(at, id, face));
        }
        if let Some(edge) = body.edges.iter().find(|e| is(&e.reference)) {
            return Some(edge_item(at, id, body, edge));
        }
        if let Some(v) = body.vertices.iter().find(|v| is(&v.reference)) {
            return Some(vertex_item(at, id, v.index, v.point));
        }
    }
    None
}
