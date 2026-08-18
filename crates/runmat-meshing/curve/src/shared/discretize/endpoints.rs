use runmat_geometry_core::{ExactBRepTopology, ExactEdge, GeometryTransform, PersistentEntityId};

use crate::shared::{SharedCurveError, SharedCurveErrorKind};

use super::edge_error;

pub(super) fn canonical_vertex_point(
    topology: &ExactBRepTopology,
    edge: &ExactEdge,
    vertex_id: &PersistentEntityId,
    transform: GeometryTransform,
) -> Result<[f64; 3], SharedCurveError> {
    let vertex = topology
        .vertices
        .binary_search_by(|vertex| vertex.id.cmp(vertex_id))
        .ok()
        .map(|index| &topology.vertices[index])
        .ok_or_else(|| {
            edge_error(
                edge,
                SharedCurveErrorKind::GeometricMismatch,
                "curve endpoint vertex",
                "exact endpoint vertex is absent from admitted topology",
            )
        })?;
    Ok(transform.transform_point(vertex.point_m))
}
