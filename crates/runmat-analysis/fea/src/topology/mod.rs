//! Immutable projection of canonical meshing topology into analysis-owned indexes and geometry.

mod boundary;
mod contract;
mod derive;
mod element;
mod flow;
mod identity;
mod point_location;
pub(crate) mod quadrature;
mod resources;

pub use contract::{
    AnalysisProjection, AnalysisProjectionError, AnalysisProjectionPolicy,
    BoundaryNormalOrientation, BoundaryQuadraturePoint, ProjectedBoundaryFace, ProjectedElement,
    ProjectedNode, VolumePointLocation, VolumeQuadraturePoint,
};
pub use flow::FlowTopologyProjection;

#[cfg(test)]
mod tests;
