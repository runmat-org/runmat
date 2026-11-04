use thiserror::Error;

use crate::assembly::elements::{
    beam::BeamElementError,
    shell::ShellElementError,
    solid::{Tetrahedron10ElementError, Tetrahedron4ElementError},
};

#[derive(Debug, Error, Clone, PartialEq)]
pub enum StructuralFieldRecoveryError {
    #[error("structural recovery numerical accounting failed: {0}")]
    Accounting(#[from] runmat_simulation_usage::LedgerError),
    #[error("structural recovery constraint projection failed: {0}")]
    ConstraintProjection(#[from] crate::constraints::ConstraintAlgebraError),
    #[error("structural field physical layout is invalid: {reason}")]
    InvalidPhysicalLayout { reason: String },
    #[error("structural recovery state has {actual} DOFs; expected {expected}")]
    DisplacementShapeMismatch { expected: usize, actual: usize },
    #[error("solid recovery has {actual} element values; expected {expected}")]
    ElementValueCountMismatch { expected: usize, actual: usize },
    #[error("physical thermoelastic recovery requires a host strain-energy-density field")]
    MissingThermoelasticEnergyDensity,
    #[error("solid thermal recovery quadrature did not converge for element {element_id}")]
    UnconvergedThermalQuadrature { element_id: String },
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
    #[error("scoped structural recovery failed: {reason}")]
    ScopedExecution { reason: String },
    #[error("structural field recovery cancelled")]
    Cancelled,
    #[error("beam recovery failed for element {element_id}: {source}")]
    Beam {
        element_id: String,
        source: BeamElementError,
    },
    #[error("shell recovery failed for element {element_id}: {source}")]
    Shell {
        element_id: String,
        source: ShellElementError,
    },
}

impl StructuralFieldRecoveryError {
    pub(super) fn geometry(
        element_id: &str,
        error: crate::assembly::SolidRecoveryGeometryError,
    ) -> Self {
        match error {
            crate::assembly::SolidRecoveryGeometryError::Tetrahedron4(source) => {
                Self::Tetrahedron4 {
                    element_id: element_id.into(),
                    source,
                }
            }
            crate::assembly::SolidRecoveryGeometryError::Tetrahedron10(source) => {
                Self::Tetrahedron10 {
                    element_id: element_id.into(),
                    source,
                }
            }
        }
    }
}
