//! Independent original boundary incidence admission.
use super::{checkpoint, invalid};
use crate::{ExactFaceMesh, ExactSurfaceBoundary, ExactSurfaceMeshError};
use runmat_geometry_core::ExactBRepTopology;
use runmat_meshing_core::MeshingCancellationSignal;
use std::collections::BTreeSet;
mod coordinates;
pub(super) fn validate_curve_boundary(
    faces: &[ExactFaceMesh],
    boundary: &ExactSurfaceBoundary,
    topology: &ExactBRepTopology,
    curves: &runmat_meshing_curve::SharedCurveMesh,
    cancellation: Option<&dyn MeshingCancellationSignal>,
) -> Result<(), ExactSurfaceMeshError> {
    for mesh in faces {
        checkpoint(cancellation)?;
        let expected_face = boundary
            .faces
            .binary_search_by(|face| face.source_face_id.cmp(&mesh.source_face_id))
            .map(|index| &boundary.faces[index])
            .map_err(|_| invalid("face result is absent from the shared curve boundary"))?;
        validate_face_curve_boundary(mesh, expected_face, topology, curves, cancellation)?;
    }
    Ok(())
}

pub(crate) fn validate_face_curve_boundary(
    mesh: &ExactFaceMesh,
    expected_face: &crate::ExactFaceBoundary,
    topology: &ExactBRepTopology,
    curves: &runmat_meshing_curve::SharedCurveMesh,
    cancellation: Option<&dyn MeshingCancellationSignal>,
) -> Result<(), ExactSurfaceMeshError> {
    let is_degenerate =
        |id: &runmat_geometry_core::PersistentEntityId| -> Result<bool, ExactSurfaceMeshError> {
            checkpoint(cancellation)?;
            topology
                .edges
                .binary_search_by(|edge| edge.id.cmp(id))
                .map(|index| topology.edges[index].is_degenerate)
                .map_err(|_| invalid("face boundary refers to an absent exact edge"))
        };
    coordinates::validate(mesh, expected_face, curves, cancellation)?;
    let mut expected_segments = Vec::new();
    for face_loop in std::iter::once(&expected_face.outer_loop).chain(&expected_face.inner_loops) {
        for segment in &face_loop.segments {
            checkpoint(cancellation)?;
            if !is_degenerate(&segment.source_edge_id)? {
                expected_segments.push(segment);
            }
        }
    }
    let expected = expected_segments
        .iter()
        .map(|segment| {
            checkpoint(cancellation)?;
            Ok((
                &segment.source_coedge_id,
                &segment.source_edge_id,
                segment.node_ids,
                segment.edge_parameters.map(f64::to_bits),
            ))
        })
        .collect::<Result<BTreeSet<_>, ExactSurfaceMeshError>>()?;
    let actual = mesh
        .boundary_segments
        .iter()
        .map(|segment| {
            checkpoint(cancellation)?;
            Ok((
                &segment.source_coedge_id,
                &segment.source_edge_id,
                segment.node_ids,
                segment.edge_parameters.map(f64::to_bits),
            ))
        })
        .collect::<Result<BTreeSet<_>, ExactSurfaceMeshError>>()?;
    if expected.len() != expected_segments.len()
        || actual.len() != mesh.boundary_segments.len()
        || actual != expected
    {
        return Err(invalid(
            "converged face boundary is stale against the current shared curve",
        ));
    }
    Ok(())
}
