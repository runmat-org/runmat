//! Cold admission binds realized shared positions to the original curve state.
use super::*;
use runmat_meshing_curve::SharedCurveMesh;

pub(super) fn validate(
    mesh: &ExactFaceMesh,
    boundary: &crate::ExactFaceBoundary,
    curves: &SharedCurveMesh,
    cancellation: Option<&dyn MeshingCancellationSignal>,
) -> Result<(), ExactSurfaceMeshError> {
    for face_loop in std::iter::once(&boundary.outer_loop).chain(&boundary.inner_loops) {
        for segment in &face_loop.segments {
            checkpoint(cancellation)?;
            let edge = curves
                .edges
                .binary_search_by(|edge| edge.source_edge_id.cmp(&segment.source_edge_id))
                .map(|index| &curves.edges[index])
                .map_err(|_| invalid("shared coordinate edge is absent from original curves"))?;
            for endpoint in 0..2 {
                checkpoint(cancellation)?;
                let parameter = segment.edge_parameters[endpoint];
                let original = edge
                    .nodes
                    .binary_search_by(|node| node.parameter.total_cmp(&parameter))
                    .map(|index| &edge.nodes[index])
                    .map_err(|_| {
                        invalid("shared coordinate parameter is absent from original curves")
                    })?;
                let node_id = segment.node_ids[endpoint];
                let actual = mesh
                    .nodes
                    .binary_search_by_key(&node_id, |node| node.node_id)
                    .map(|index| &mesh.nodes[index])
                    .map_err(|_| invalid("shared boundary coordinate is absent from face mesh"))?;
                if original.node_id != node_id
                    || original.coordinates_m.canonical_value() != actual.point_m
                {
                    return Err(invalid(
                        "realized shared boundary coordinate differs from the original curve state",
                    )
                    .with_face(&mesh.source_face_id));
                }
            }
        }
    }
    Ok(())
}
