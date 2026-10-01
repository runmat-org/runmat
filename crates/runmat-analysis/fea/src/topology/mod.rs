//! Immutable projection of canonical meshing topology into analysis-owned indexes and geometry.

mod boundary;
mod contract;
mod derive;
mod edge_space;
mod element;
mod flow;
mod identity;
mod point_location;
pub(crate) mod quadrature;
mod resources;
#[cfg(test)]
pub(crate) mod test_meshes;

pub use contract::{
    AnalysisProjection, AnalysisProjectionError, AnalysisProjectionPolicy,
    BoundaryNormalOrientation, BoundaryQuadraturePoint, ProjectedBoundaryFace, ProjectedElement,
    ProjectedNode, VolumePointLocation, VolumeQuadraturePoint,
};
pub use edge_space::{EdgeElementMoments, EdgeQuadraturePoint, OrientedEdge, TetrahedralEdgeSpace};
pub use flow::FlowTopologyProjection;

#[cfg(test)]
mod tests;
