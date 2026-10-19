//! Explicit affine-geometry eligibility; no physical coefficient assembly or solver.
use super::{topology, MaxwellError};
use crate::topology::{AnalysisProjection, TetrahedralEdgeSpace};
use runmat_meshing_core::ElementOrder;

pub(super) fn spaces(
    projection: &AnalysisProjection,
) -> Result<(TetrahedralEdgeSpace, Vec<usize>), MaxwellError> {
    let length = projection.characteristic_length_m;
    if !length.is_finite() || length <= 0.0 || projection.elements.is_empty() {
        return Err(topology("empty or invalid Maxwell geometry"));
    }
    for element in &projection.elements {
        if element.order != ElementOrder::Tet10 {
            continue;
        }
        for (midside, [a, b]) in runmat_meshing_core::TETRAHEDRON_MIDSIDE_EDGE_CORNERS
            .into_iter()
            .enumerate()
        {
            let middle = projection.nodes[element.node_indices[4 + midside]].coordinates_m;
            let left = projection.nodes[element.node_indices[a]].coordinates_m;
            let right = projection.nodes[element.node_indices[b]].coordinates_m;
            if (0..3).any(|axis| {
                (middle[axis] - 0.5 * (left[axis] + right[axis])).abs()
                    > 64.0 * f64::EPSILON * length
            }) {
                return Err(MaxwellError::Unsupported(
                    "curved Maxwell form quadrature is not qualified; Tet10 must be affine".into(),
                ));
            }
        }
    }
    let edges = TetrahedralEdgeSpace::derive(projection).map_err(topology)?;
    let exterior = edges.exterior_edge_indices(projection).map_err(topology)?;
    Ok((edges, exterior))
}
