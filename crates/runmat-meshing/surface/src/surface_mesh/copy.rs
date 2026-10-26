//! Cooperatively copy prerequisite inventories without changing their values or order.
use super::{control::checkpoint, ExactSurfaceMeshError};
use crate::{ExactFaceMesh, ExactFaceMeshNode, ExactFaceMeshNodeUse};
use runmat_meshing_core::MeshingCancellationSignal;

pub(super) fn clone_items<T: Clone>(
    items: &[T],
    control: Option<&dyn MeshingCancellationSignal>,
) -> Result<Vec<T>, ExactSurfaceMeshError> {
    let mut copied = Vec::with_capacity(items.len());
    for item in items {
        checkpoint(control)?;
        copied.push(item.clone());
    }
    Ok(copied)
}

pub(super) fn clone_faces(
    faces: &[ExactFaceMesh],
    control: Option<&dyn MeshingCancellationSignal>,
) -> Result<Vec<ExactFaceMesh>, ExactSurfaceMeshError> {
    let mut copied = Vec::with_capacity(faces.len());
    for face in faces {
        checkpoint(control)?;
        let mut nodes = Vec::with_capacity(face.nodes.len());
        for node in &face.nodes {
            checkpoint(control)?;
            let mut uses = Vec::with_capacity(node.uses.len());
            for use_record in &node.uses {
                checkpoint(control)?;
                uses.push(ExactFaceMeshNodeUse {
                    source_face_id: use_record.source_face_id.clone(),
                    chart_id: use_record.chart_id,
                    uv: use_record.uv,
                    evaluator_uv: use_record.evaluator_uv,
                    exact_edge_parameters: clone_items(&use_record.exact_edge_parameters, control)?,
                });
            }
            nodes.push(ExactFaceMeshNode {
                node_id: node.node_id,
                point_m: node.point_m,
                uses,
            });
        }
        copied.push(ExactFaceMesh {
            schema_version: face.schema_version,
            source_face_id: face.source_face_id.clone(),
            nodes,
            triangles: clone_items(&face.triangles, control)?,
            boundary_segments: clone_items(&face.boundary_segments, control)?,
            joined_chart_cuts: clone_items(&face.joined_chart_cuts, control)?,
            maximum_chordal_deviation_m: runmat_types::physical::Length::new(
                face.maximum_chordal_deviation_m.canonical_value(),
            ),
            maximum_normal_deviation_rad: face.maximum_normal_deviation_rad,
        });
    }
    Ok(copied)
}
