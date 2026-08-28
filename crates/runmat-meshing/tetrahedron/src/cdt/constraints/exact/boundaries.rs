use std::collections::BTreeMap;

use runmat_geometry_core::{ExactBRepTopology, PersistentEntityId};
use runmat_meshing_core::StableDigest;
use runmat_meshing_surface::ExactSurfaceMesh;

use super::{invalid_boundary, sorted_id_pair, DelaunayConstraintError};

#[derive(Clone, Debug, PartialEq)]
pub(super) struct ExactBoundarySegment {
    pub(super) edge_id: PersistentEntityId,
    pub(super) parameters: [f64; 2],
}

pub(super) fn boundary_edges(
    topology: &ExactBRepTopology,
    surface: &ExactSurfaceMesh,
) -> Result<BTreeMap<[StableDigest; 2], ExactBoundarySegment>, DelaunayConstraintError> {
    let coedges = topology
        .coedges
        .iter()
        .map(|coedge| (&coedge.id, coedge))
        .collect::<BTreeMap<_, _>>();
    let mut result = BTreeMap::new();
    for segment in &surface.boundary_segments {
        let coedge = coedges.get(&segment.source_coedge_id).ok_or_else(|| {
            invalid_boundary("surface boundary segment references an absent exact coedge")
        })?;
        if segment.source_edge_id != coedge.edge_id
            || segment.node_ids[0] == segment.node_ids[1]
            || !segment
                .edge_parameters
                .iter()
                .all(|value| value.is_finite())
            || segment.edge_parameters[0] == segment.edge_parameters[1]
        {
            return Err(invalid_boundary(
                "surface boundary segment must match its exact edge and have distinct nodes and parameters",
            ));
        }
        let key = sorted_id_pair(segment.node_ids);
        let parameters = if key == segment.node_ids {
            segment.edge_parameters
        } else {
            [segment.edge_parameters[1], segment.edge_parameters[0]]
        };
        let exact = ExactBoundarySegment {
            edge_id: coedge.edge_id.clone(),
            parameters,
        };
        if result
            .insert(key, exact.clone())
            .is_some_and(|previous| previous != exact)
        {
            return Err(invalid_boundary(
                "one exact surface segment maps to conflicting persistent edges",
            ));
        }
    }
    Ok(result)
}
