//! `sketch.use-edges` in a part: the part's own edges and faces brought into the open sketch.
//!
//! Each new entity is written with where it comes from (`from = project(<edge>)`,
//! `silhouette(<face>, chain = n)`, `intersection(<face>, chain = n)` or
//! `offset_on_surface(<edge>, face = <face>, distance = d)`), and with its first position,
//! read from the regenerated part's display by the same functions the regeneration uses
//! ([`runmat_cad::sketch_conversion`]). On every rebuild the part reads it again from its
//! bodies and holds it fixed, so it follows the model.
//!
//! - **Project** takes edges, faces (each edge round the face) or bodies (every edge).
//! - **Intersection** takes faces or bodies (every face that crosses the plane): one entity
//!   per chain of the cut.
//! - **Silhouette** takes curved faces or bodies (every curved face with an outline).
//! - **Offset on surface** takes a planar face (each edge round it, offset into it) or edges
//!   with the planar face they cross: a face picked with them, else the edge's planar face
//!   that lies square to the sketch normal.

use super::super::{model, EditFailure, SourceChange};
use super::offset_edges::{fresh, stem, Converted};
use crate::features::value;
use runmat_cad::display::{BodyDisplay, FaceDisplay, PartDisplay};
use runmat_cad::model::datum::Frame;
use runmat_cad::model::sketch::{SketchEntity, SketchProjection};
use runmat_cad::model::FeatureKind;
use runmat_cad::refs::history::{EdgeCurve, SurfaceKind};
use runmat_cad::refs::reference::Reference;
use runmat_cad::regen::kernel::FaceMesh;
use runmat_cad::sketch_conversion::{
    fit_curve, in_plane, join_offset_corners, offset_on_face, plane_section, silhouette, Curve,
    FaceSurface,
};
use runmat_cad::syntax::edit::edit_part;
use runmat_cad::value::ValueExpr;
use serde_json::{Map, Value};

type Values = Map<String, Value>;

fn f64s3(p: &[f32]) -> Vec<[f64; 3]> {
    p.chunks_exact(3)
        .map(|c| [c[0] as f64, c[1] as f64, c[2] as f64])
        .collect()
}

fn face_mesh(face: &FaceDisplay) -> FaceMesh {
    FaceMesh {
        positions: f64s3(&face.positions),
        triangles: face
            .triangles
            .chunks_exact(3)
            .map(|t| [t[0], t[1], t[2]])
            .collect(),
        normals: f64s3(&face.normals),
    }
}

/// What a pick names on the regenerated part.
enum Picked<'d> {
    Edge(&'d BodyDisplay, u32),
    Face(&'d BodyDisplay, &'d FaceDisplay),
    Body(&'d BodyDisplay),
}

fn find<'d>(display: &'d PartDisplay, text: &str) -> Option<Picked<'d>> {
    find_written(display, text).or_else(|| {
        // The same reference written another way (argument order, spacing): compare the
        // lowered references.
        let wanted = reference(text).ok()?;
        let same = |r: &Option<String>| {
            r.as_deref().and_then(|t| reference(t).ok()) == Some(wanted.clone())
        };
        for body in &display.bodies {
            if same(&body.reference) {
                return Some(Picked::Body(body));
            }
            if let Some(e) = body.edges.iter().find(|e| same(&e.reference)) {
                return Some(Picked::Edge(body, e.index));
            }
            if let Some(f) = body.faces.iter().find(|f| same(&f.reference)) {
                return Some(Picked::Face(body, f));
            }
        }
        None
    })
}

fn find_written<'d>(display: &'d PartDisplay, text: &str) -> Option<Picked<'d>> {
    for body in &display.bodies {
        if body.reference.as_deref() == Some(text) {
            return Some(Picked::Body(body));
        }
        if let Some(e) = body
            .edges
            .iter()
            .find(|e| e.reference.as_deref() == Some(text))
        {
            return Some(Picked::Edge(body, e.index));
        }
        if let Some(f) = body
            .faces
            .iter()
            .find(|f| f.reference.as_deref() == Some(text) || f.names.iter().any(|n| n == text))
        {
            return Some(Picked::Face(body, f));
        }
    }
    None
}

fn reference(text: &str) -> Result<Reference, EditFailure> {
    serde_json::from_value(crate::picks::lower_text(text)?)
        .map_err(|e| model(format!("`{text}` isn't a reference: {e}")))
}

/// One entity to add: where it comes from and its first position.
struct New {
    from: SketchProjection,
    curve: Curve,
}

fn edge_points(body: &BodyDisplay, edge: u32) -> Vec<[f64; 3]> {
    body.edges
        .iter()
        .find(|e| e.index == edge)
        .map(|e| f64s3(&e.points))
        .unwrap_or_default()
}

fn edge_ref(body: &BodyDisplay, edge: u32) -> Result<Reference, EditFailure> {
    let text = body
        .edges
        .iter()
        .find(|e| e.index == edge)
        .and_then(|e| e.reference.clone())
        .ok_or_else(|| model("the edge has no name in the part to follow"))?;
    reference(&text)
}

fn face_ref(face: &FaceDisplay) -> Result<Reference, EditFailure> {
    let text = face
        .reference
        .clone()
        .ok_or_else(|| model("the face has no name in the part to follow"))?;
    reference(&text)
}

/// The edges round `face`.
fn edges_of(body: &BodyDisplay, face: &FaceDisplay) -> Vec<u32> {
    body.edges
        .iter()
        .filter(|e| e.faces.contains(&face.index) && e.reference.is_some())
        .map(|e| e.index)
        .collect()
}

fn curve_of(frame: &Frame, points: &[[f64; 3]], what: &str) -> Result<Curve, EditFailure> {
    let local: Vec<[f64; 2]> = points.iter().map(|p| in_plane(frame, *p)).collect();
    fit_curve(&local)
        .ok_or_else(|| model(format!("{what} projects to a point on this sketch's plane")))
}

fn project(frame: &Frame, body: &BodyDisplay, edge: u32) -> Result<New, EditFailure> {
    Ok(New {
        from: SketchProjection::Project {
            edge: edge_ref(body, edge)?,
        },
        curve: curve_of(frame, &edge_points(body, edge), "the edge")?,
    })
}

/// Silhouette or intersection chains of one face; `strict` refuses a face without any.
fn chains(
    frame: &Frame,
    face: &FaceDisplay,
    outline: bool,
    strict: bool,
) -> Result<Vec<New>, EditFailure> {
    let mesh = face_mesh(face);
    let surface = FaceSurface {
        mesh: &mesh,
        surface: face.surface,
    };
    let found = if outline {
        silhouette(&surface, frame)
    } else {
        plane_section(&surface, frame)
    };
    let all = match found {
        Ok(all) => all,
        Err(_) if !strict => return Ok(Vec::new()),
        Err(m) => {
            return Err(model(format!(
                "{}: {m}",
                face.label.clone().unwrap_or_else(|| "the face".into())
            )))
        }
    };
    let r = face_ref(face)?;
    let mut out = Vec::new();
    for (chain, points) in all.iter().enumerate() {
        let from = if outline {
            SketchProjection::Silhouette {
                face: r.clone(),
                chain: chain as u32,
            }
        } else {
            SketchProjection::Intersection {
                face: r.clone(),
                chain: chain as u32,
            }
        };
        out.push(New {
            from,
            curve: curve_of(frame, points, "the curve")?,
        });
    }
    Ok(out)
}

/// The planar face an edge pick is offset across: the one picked with it, else its planar
/// face square to the sketch normal, else its only planar face.
fn offset_face<'d>(
    frame: &Frame,
    body: &'d BodyDisplay,
    edge: u32,
    picked: Option<&'d FaceDisplay>,
) -> Result<&'d FaceDisplay, EditFailure> {
    if let Some(face) = picked {
        return Ok(face);
    }
    let e = body
        .edges
        .iter()
        .find(|e| e.index == edge)
        .ok_or_else(|| model("the edge isn't drawn"))?;
    let planar: Vec<&FaceDisplay> = e
        .faces
        .iter()
        .filter_map(|i| body.faces.iter().find(|f| f.index == *i))
        .filter(|f| matches!(f.surface, Some(SurfaceKind::Plane { .. })))
        .collect();
    let square = planar.iter().copied().find(|f| match f.surface {
        Some(SurfaceKind::Plane { normal }) => {
            let z = frame.z;
            let d = normal[0] * z[0] + normal[1] * z[1] + normal[2] * z[2];
            d.abs() > 0.999
        }
        _ => false,
    });
    square
        .or_else(|| (planar.len() == 1).then(|| planar[0]))
        .ok_or_else(|| model("pick the planar face to offset the edge across along with it"))
}

/// The regenerated part, for reading the edges a sketch brings in.
fn compile(path: &str, text: &str) -> Result<crate::runtime::CompiledPart, EditFailure> {
    crate::runtime::compile_part(path, text).map_err(|d| {
        model(format!(
            "the part doesn't compile, so nothing can be brought into the sketch: {}",
            d.first().map(|d| d.message.as_str()).unwrap_or("")
        ))
    })
}

/// The drawn frame of sketch `sketch_id` (part coordinates, metres).
fn sketch_frame(display: &PartDisplay, sketch_id: &str) -> Result<Frame, EditFailure> {
    let drawn = display
        .sketches
        .iter()
        .find(|s| s.id == sketch_id)
        .ok_or_else(|| {
            model(format!(
                "`{sketch_id}` isn't drawn, so nothing can be projected onto it"
            ))
        })?;
    let v = |a: [f32; 3]| [a[0] as f64, a[1] as f64, a[2] as f64];
    Ok(Frame {
        origin: v(drawn.frame.origin),
        x: v(drawn.frame.x),
        y: v(drawn.frame.y),
        z: v(drawn.frame.z),
    })
}

/// Whether `pick` names an edge of the part's model (rather than a sketch curve or point).
pub(super) fn is_model_edge(pick: &str) -> bool {
    !pick.starts_with("sketch.")
        && crate::picks::lower_text(pick).is_ok_and(|r| crate::picks::topo(&r) == "edge")
}

/// Each picked model edge projected onto sketch `sketch_id`, as Project would bring it in.
pub(super) fn project_edges(
    path: &str,
    text: &str,
    sketch_id: &str,
    picks: &[String],
) -> Result<Vec<Converted>, EditFailure> {
    let compiled = compile(path, text)?;
    let display = &compiled.display;
    let frame = sketch_frame(display, sketch_id)?;
    picks
        .iter()
        .map(|p| match find(display, p) {
            Some(Picked::Edge(body, e)) => {
                let n = project(&frame, body, e)?;
                Ok((n.from, n.curve.geometry()))
            }
            _ => Err(model(format!("`{p}` isn't an edge of the part"))),
        })
        .collect()
}

/// Brings the picked edges and faces into the open sketch.
pub(super) fn use_edges(
    path: &str,
    text: &str,
    values: &Values,
) -> Result<(SourceChange, String), EditFailure> {
    let sketch_id = values
        .get("sketch")
        .and_then(Value::as_str)
        .ok_or_else(|| model("open a sketch first"))?
        .to_owned();
    let kind = values
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or("project");
    let items = super::picks(values, "items");
    if items.is_empty() {
        return Err(model(match kind {
            "intersection" => "pick the faces or bodies to cut with the sketch plane",
            "silhouette" => "pick the curved faces or bodies to outline",
            "offset_on_surface" => {
                "pick a planar face, or edges with the face to offset them across"
            }
            _ => "pick the edges, faces or bodies to project into the sketch",
        }));
    }
    let compiled = compile(path, text)?;
    let display = &compiled.display;
    let frame = sketch_frame(display, &sketch_id)?;
    if kind == "intersection" || kind == "silhouette" {
        for p in &items {
            if crate::picks::lower_text(p).is_ok_and(|r| crate::picks::topo(&r) == "edge") {
                return Err(model(format!(
                    "{} reads faces; pick faces or a body, not an edge",
                    if kind == "silhouette" {
                        "Silhouette"
                    } else {
                        "Intersection"
                    }
                )));
            }
        }
    }
    let picked: Vec<Picked<'_>> = items
        .iter()
        .map(|p| {
            find(display, p)
                .ok_or_else(|| model(format!("`{p}` isn't an edge, face or body of the part")))
        })
        .collect::<Result<_, _>>()?;
    let mut new: Vec<New> = Vec::new();
    match kind {
        "project" => {
            for p in &picked {
                match p {
                    Picked::Edge(body, e) => new.push(project(&frame, body, *e)?),
                    Picked::Face(body, face) => {
                        for e in edges_of(body, face) {
                            new.push(project(&frame, body, e)?);
                        }
                    }
                    Picked::Body(body) => {
                        for e in body.edges.iter().filter(|e| e.reference.is_some()) {
                            // An edge seen end-on projects to a point and is left out.
                            if let Ok(n) = project(&frame, body, e.index) {
                                new.push(n);
                            }
                        }
                    }
                }
            }
        }
        "intersection" | "silhouette" => {
            let outline = kind == "silhouette";
            for p in &picked {
                match p {
                    Picked::Face(_, face) => new.extend(chains(&frame, face, outline, true)?),
                    Picked::Body(body) => {
                        for face in &body.faces {
                            new.extend(chains(&frame, face, outline, false)?);
                        }
                    }
                    Picked::Edge(..) => {
                        return Err(model(format!(
                            "{} reads faces; pick faces or a body, not an edge",
                            if outline {
                                "Silhouette"
                            } else {
                                "Intersection"
                            }
                        )))
                    }
                }
            }
        }
        "offset_on_surface" => {
            let distance = value(values, "distance", "mm")?;
            let d_mm = distance["value"].as_f64().unwrap_or(1.0);
            let distance_expr: ValueExpr = serde_json::from_value(distance.clone())
                .map_err(|e| model(format!("the offset: {e}")))?;
            let faces: Vec<(&BodyDisplay, &FaceDisplay)> = picked
                .iter()
                .filter_map(|p| match p {
                    Picked::Face(b, f) => Some((*b, *f)),
                    _ => None,
                })
                .collect();
            let edges: Vec<(&BodyDisplay, u32)> = picked
                .iter()
                .filter_map(|p| match p {
                    Picked::Edge(b, e) => Some((*b, *e)),
                    _ => None,
                })
                .collect();
            if picked.iter().any(|p| matches!(p, Picked::Body(_))) {
                return Err(model(
                    "Offset on surface moves edges across a face; pick a face or edges, not a body",
                ));
            }
            // (body, edge, face) to offset.
            let mut jobs: Vec<(&BodyDisplay, u32, &FaceDisplay)> = Vec::new();
            if edges.is_empty() {
                for (body, face) in &faces {
                    for e in edges_of(body, face) {
                        jobs.push((*body, e, *face));
                    }
                }
            } else {
                if faces.len() > 1 {
                    return Err(model("pick edges with one face to offset them across"));
                }
                for (body, e) in &edges {
                    let face = offset_face(&frame, body, *e, faces.first().map(|f| f.1))?;
                    jobs.push((*body, *e, face));
                }
            }
            let mut offsets: Vec<(Curve, [[f64; 2]; 2])> = Vec::new();
            let mut sources = Vec::new();
            for (body, e, face) in &jobs {
                let points = edge_points(body, *e);
                if points.len() < 2 {
                    return Err(model("the edge isn't drawn"));
                }
                let mesh = face_mesh(face);
                let moved = offset_on_face(
                    &points,
                    EdgeCurve::Other,
                    &FaceSurface {
                        mesh: &mesh,
                        surface: face.surface,
                    },
                    d_mm / 1000.0,
                )
                .map_err(model)?;
                let ends = [
                    in_plane(&frame, points[0]),
                    in_plane(&frame, points[points.len() - 1]),
                ];
                offsets.push((curve_of(&frame, &moved, "the offset edge")?, ends));
                sources.push(SketchProjection::OffsetOnSurface {
                    edge: edge_ref(body, *e)?,
                    face: face_ref(face)?,
                    distance: distance_expr.clone(),
                });
            }
            join_offset_corners(&mut offsets);
            for ((curve, _), from) in offsets.into_iter().zip(sources) {
                new.push(New { from, curve });
            }
        }
        other => return Err(model(format!("Use edges has no type `{other}`"))),
    }
    if new.is_empty() {
        return Err(model("nothing picked gives a curve on this sketch's plane"));
    }
    let mut added: Vec<String> = Vec::new();
    let change = edit_part(text, path, |document| {
        let sketch = match document.feature_mut(&sketch_id).map(|f| &mut f.kind) {
            Some(FeatureKind::Sketch(s)) => s,
            _ => return Err(model(format!("the part has no sketch `{sketch_id}`"))),
        };
        for n in &new {
            let geometry = n.curve.geometry();
            let id = fresh(sketch, stem(&geometry));
            added.push(id.0.clone());
            sketch.entities.push(SketchEntity {
                id,
                geometry,
                construction: false,
                projected: Some(n.from.clone()),
            });
        }
        Ok(())
    })?;
    Ok((
        change,
        format!("Brought {} into {sketch_id}", added.join(", ")),
    ))
}
