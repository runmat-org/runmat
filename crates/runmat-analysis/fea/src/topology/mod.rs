//! Immutable projection of canonical meshing topology into analysis-owned indexes and geometry.

mod boundary;
mod contract;
mod derive;
mod edge_space;
mod element;
mod finite_volume;
mod flow;
mod identity;
mod point_location;
pub(crate) mod quadrature;
mod resources;
mod solid_integration;
pub(crate) mod storage;
#[cfg(test)]
pub(crate) mod test_meshes;

pub use contract::{
    AnalysisProjection, AnalysisProjectionError, AnalysisProjectionPolicy,
    BoundaryNormalOrientation, BoundaryQuadraturePoint, ProjectedBoundaryFace, ProjectedElement,
    ProjectedNode, VolumePointLocation, VolumeQuadraturePoint,
};
pub use edge_space::{EdgeElementMoments, EdgeQuadraturePoint, OrientedEdge, TetrahedralEdgeSpace};
pub use finite_volume::{
    FiniteVolumeBoundarySource, FiniteVolumeCell, FiniteVolumeFace, FiniteVolumeIncidence,
    FiniteVolumeProjectionError, FiniteVolumeProjectionPolicy, FiniteVolumeReservation,
    FiniteVolumeTopology,
};
pub use flow::FlowTopologyProjection;
pub use solid_integration::{
    CanonicalSolidIntegrationElement, CanonicalSolidIntegrationPlan, CanonicalSolidIntegrationRule,
    SolidIntegrationPlanError,
};

#[cfg(test)]
mod tests;
