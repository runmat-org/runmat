//! Immutable projection of canonical meshing topology into analysis-owned indexes and geometry.

mod boundary;
mod contract;
mod derive;
pub(crate) use derive::ProjectionCancellation;
mod edge_space;
mod element;
mod finite_volume;
mod flow;
mod identity;
mod neighbors;
mod point_location;
pub(crate) mod quadrature;
mod resources;
mod solid_integration;
pub(crate) mod storage;
mod surface_search;
#[cfg(test)]
pub(crate) mod test_meshes;

pub use contract::{
    AnalysisProjection, AnalysisProjectionError, AnalysisProjectionPolicy,
    BoundaryNormalOrientation, BoundaryQuadraturePoint, ProjectedBoundaryFace, ProjectedElement,
    ProjectedNode, VolumePointLocation, VolumeQuadraturePoint,
};
pub use edge_space::{EdgeElementMoments, EdgeQuadraturePoint, OrientedEdge, TetrahedralEdgeSpace};
pub use finite_volume::{
    FiniteVolumeBoundarySource, FiniteVolumeCell, FiniteVolumeCellSelection, FiniteVolumeFace,
    FiniteVolumeIncidence, FiniteVolumeProjectionError, FiniteVolumeProjectionPolicy,
    FiniteVolumeReservation, FiniteVolumeTopology,
};
pub use flow::FlowTopologyProjection;

/// Visit the canonical volume quadrature of one isolated tetrahedron.
///
/// This is the rule every volume operator of the shared projection integrates with; element
/// kernels that run without a whole projection (one subdomain of a decomposed mesh) use it so
/// their element operators equal the assembled ones exactly. Returns `None` for a degenerate or
/// inverted element or a node count that differs from `order`.
pub fn visit_tetrahedron_volume_quadrature(
    order: runmat_meshing_core::ElementOrder,
    coordinates: &[[f64; 3]],
    visit: impl FnMut(VolumeQuadraturePoint),
) -> Result<Option<()>, runmat_simulation_usage::LedgerError> {
    quadrature::visit_volume_quadrature(order, coordinates, visit)
}
#[cfg(test)]
pub(crate) use solid_integration::curved_rule as solid_integration_curved_rule;
pub use solid_integration::{
    CanonicalSolidIntegrationElement, CanonicalSolidIntegrationPlan, CanonicalSolidIntegrationRule,
    SolidIntegrationPlanError,
};

#[cfg(test)]
mod tests;

#[cfg(test)]
pub(crate) use resources::measured as measure_test_allocations;

pub use surface_search::{
    BoundaryPairingError, BoundaryPairingPolicy, BoundaryProjectionDifferentialPolicy,
    BoundaryProjectionDirection, BoundaryProjectionMixedVariation, BoundaryProjectionVariation,
    BoundarySearchError, BoundarySearchPolicy, BoundaryStationaryPolicy, BoundarySurface,
    BoundarySurfaceBounds, BoundarySurfaceIndex, BoundarySurfaceMatch, BoundarySurfacePoint,
    ClosestBoundaryPoint, OrientedBoundarySurface, StationaryBoundaryProjection,
};

/// Node-based (one DOF per projected node) element incidence for parallel element assembly.
pub(crate) fn projection_element_map(
    projection: &AnalysisProjection,
) -> Result<
    crate::algebra::element_assembly::ElementDofMap,
    crate::algebra::element_assembly::ElementAssemblyError<std::convert::Infallible>,
> {
    let mut offsets = Vec::with_capacity(projection.elements.len() + 1);
    offsets.push(0);
    let mut dofs = Vec::with_capacity(
        projection
            .elements
            .iter()
            .map(|e| e.node_indices.len())
            .sum(),
    );
    for element in &projection.elements {
        for &node in &element.node_indices {
            dofs.push(u32::try_from(node).map_err(|_| {
                crate::algebra::element_assembly::ElementAssemblyError::DofIndexOverflow(node)
            })?);
        }
        offsets.push(dofs.len());
    }
    crate::algebra::element_assembly::ElementDofMap::new(projection.nodes.len(), offsets, dofs)
}
