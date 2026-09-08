//! Immutable projection of canonical meshing topology into analysis-owned indexes and geometry.

mod boundary;
mod contract;
mod derive;
mod element;
mod identity;
mod quadrature;

pub use contract::{
    AnalysisProjection, AnalysisProjectionError, AnalysisProjectionPolicy,
    BoundaryNormalOrientation, BoundaryQuadraturePoint, ProjectedBoundaryFace, ProjectedElement,
    ProjectedNode, VolumeQuadraturePoint,
};

#[cfg(test)]
mod tests;
