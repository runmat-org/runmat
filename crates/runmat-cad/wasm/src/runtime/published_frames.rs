//! Single faces and vertices a part publishes, as frames an assembly's mate connectors anchor to
//! (`assembly.mate` publishes the faces, edges and vertices it mates). A face gives the frame the
//! assembly's mate picks read from its display ([`face_frame`]): a planar face a plane, a round
//! one a frame on its axis; a vertex gives its point. A single published edge is a one-chain path
//! ([`super::published_paths`]), whose frame the assembly reads from the chain.

use super::RegeneratedPart;
use crate::edit::assembly::frames::{cross, face_frame, FrameKind};
use runmat_cad::assembly::source::PublishedGeometry;
use runmat_cad::ids::PublishName;
use runmat_cad::model::datum::Frame;
use runmat_cad::refs::name::TopoKind;
use runmat_cad::refs::resolve::EntityHandle;
use std::collections::BTreeMap;

/// Every publish of `part` that names exactly one face or vertex, as its frame.
pub(super) fn published_frames(part: &RegeneratedPart) -> BTreeMap<PublishName, PublishedGeometry> {
    let mut out = BTreeMap::new();
    for (name, target) in &part.result.publishes {
        let Ok(target) = target else { continue };
        let [entity] = target.entities.as_slice() else {
            continue;
        };
        if !matches!(entity.kind, TopoKind::Face | TopoKind::Vertex) {
            continue;
        }
        let found = part
            .result
            .state
            .bodies
            .iter()
            .enumerate()
            .find_map(|(b, body)| {
                let count = match entity.kind {
                    TopoKind::Face => body.topology.faces.len(),
                    _ => body.topology.vertices.len(),
                };
                let index = (0..count).find(|&i| {
                    let names = part.result.state.entity_names(&EntityHandle {
                        body: b as u32,
                        kind: entity.kind,
                        index: i as u32,
                    });
                    !names.is_empty() && names == entity.names
                })?;
                let display = part.display.bodies.iter().find(|d| d.index as usize == b)?;
                match entity.kind {
                    TopoKind::Face => {
                        let face = display.faces.iter().find(|f| f.index as usize == index)?;
                        let f = face_frame(&serde_json::to_value(face).ok()?)?;
                        let frame = Frame {
                            origin: f.origin,
                            x: f.x,
                            y: cross(f.z, f.x),
                            z: f.z,
                        };
                        Some(match f.kind {
                            FrameKind::Plane => PublishedGeometry::Plane { frame },
                            _ => PublishedGeometry::Frame { frame },
                        })
                    }
                    _ => {
                        let vertex = display
                            .vertices
                            .iter()
                            .find(|v| v.index as usize == index)?;
                        Some(PublishedGeometry::Point {
                            at: vertex.point.map(f64::from),
                        })
                    }
                }
            });
        if let Some(geometry) = found {
            out.insert(name.clone(), geometry);
        }
    }
    out
}
