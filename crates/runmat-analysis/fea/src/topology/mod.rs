//! Immutable projection of canonical meshing topology into analysis-owned indexes and geometry.

mod boundary;
mod contract;
mod derive;
mod element;
mod identity;

pub use contract::{
    AnalysisProjection, AnalysisProjectionError, AnalysisProjectionPolicy,
    BoundaryNormalOrientation, ProjectedBoundaryFace, ProjectedElement, ProjectedNode,
};

#[cfg(test)]
mod tests;
