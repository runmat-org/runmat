use thiserror::Error;

use crate::assembly::elements::{
    shell::ShellElementError,
    solid::{Tetrahedron10ElementError, Tetrahedron4ElementError},
};

#[derive(Debug, Error, Clone, PartialEq)]
pub enum StructuralFieldRecoveryError {
    #[error("structural recovery state has {actual} DOFs; expected {expected}")]
    DisplacementShapeMismatch { expected: usize, actual: usize },
    #[error("solid recovery has {actual} element values; expected {expected}")]
    ElementValueCountMismatch { expected: usize, actual: usize },
    #[error("solid recovery element {element_id} references absent node {node_index}")]
    InvalidNodeIndex {
        element_id: String,
        node_index: usize,
    },
    #[error("solid recovery displacement is absent or non-finite at node {node_index}")]
    InvalidDisplacement { node_index: usize },
    #[error("Tet4 solid recovery failed for element {element_id}: {source}")]
    Tetrahedron4 {
        element_id: String,
        source: Tetrahedron4ElementError,
    },
    #[error("Tet10 solid recovery failed for element {element_id}: {source}")]
    Tetrahedron10 {
        element_id: String,
        source: Tetrahedron10ElementError,
    },
    #[error("solid recovery material is invalid for element {element_id}")]
    InvalidMaterial { element_id: String },
    #[error("solid recovery produced a non-finite field for element {element_id}")]
    NonFiniteResult { element_id: String },
    #[error("shell recovery failed for element {element_id}: {source}")]
    Shell {
        element_id: String,
        source: ShellElementError,
    },
}
