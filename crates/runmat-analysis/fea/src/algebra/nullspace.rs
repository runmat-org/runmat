use std::collections::BTreeSet;

use runmat_canonical_codec::CanonicalLimits;
use runmat_execution::Digest;
use runmat_meshing_core::{SolverMeshArtifact, StableDigest};
use serde::Serialize;
use thiserror::Error;

use crate::{
    assembly::dofs::StructuralDofLayout,
    constraints::{CanonicalStructuralConstraintPlan, ResolvedConstraintMatrix},
    topology::AnalysisProjection,
};

use super::{spmv, MatrixDefiniteness, MatrixSymmetry, OperatorRole, RealSparseMatrix};

mod canonical_frame;
mod components;
mod mixed_structural;
mod rigid_body;
mod scalar;
mod scalar_gauge;
pub(crate) use scalar_gauge::prepare_scalar_component_gauges;
mod solid;
mod solid_components;
mod structural_frame;

pub(crate) use components::connected_components;

const EVIDENCE_PREFIX: &[u8] = b"runmat-fea-nullspace-evidence/v1\0";
const EVIDENCE_DOMAIN: &str = "fea-nullspace-evidence";
const BINDING_DOMAIN: &str = "fea-prepared-nullspace-binding";
pub(super) const CONSTRAINT_DOMAIN: &str = "fea-nullspace-constraint-state";
const RHS_DOMAIN: &str = "fea-nullspace-prepared-rhs";
const LIMITS: CanonicalLimits = CanonicalLimits::new(64 * 1024 * 1024, 1_000_000, 65_536, 64);
const RANK_TOLERANCE: f64 = 1.0e-10;
const NULL_MODE_RELATIVE_TOLERANCE: f64 = 1.0e-9;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum NullspaceKind {
    ScalarConstantPerComponent,
    SolidRigidBody3d,
    StructuralFrameRigidBody3d,
    MixedStructuralRigidBody3d,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct NullspaceComponentEvidence {
    component_index: usize,
    node_identities: Vec<StableDigest>,
    null_mode_count: usize,
    projected_anchor_rank: usize,
    anchoring: NullspaceAnchoring,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum NullspaceAnchoring {
    PrescribedDof {
        count: usize,
    },
    PositiveRobin {
        count: usize,
    },
    PrescribedAndPositiveRobin {
        prescribed_count: usize,
        robin_count: usize,
    },
    ConstraintRowSpace,
    OperatorAndConstraintRowSpace,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct NullspaceEliminationEvidence {
    schema_version: u16,
    kind: NullspaceKind,
    topology_identity: Digest,
    operator_pattern_identity: Digest,
    operator_values_identity: Digest,
    constraint_identity: Digest,
    constrained_dofs_identity: Digest,
    components: Vec<NullspaceComponentEvidence>,
    proof_identity: Digest,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct PreparedNullspaceEvidence {
    source: NullspaceEliminationEvidence,
    prepared_pattern_identity: Digest,
    prepared_values_identity: Digest,
    prepared_rhs_identity: Digest,
    binding_identity: Digest,
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum NullspaceEvidenceError {
    #[error("nullspace evidence dimensions do not match the physical topology")]
    DimensionMismatch,
    #[error("a connected component retains an uneliminated null mode")]
    UnderconstrainedComponent,
    #[error("the original component source is incompatible with its constant null mode")]
    IncompatibleConstantModeSource,
    #[error("a reference point cannot be assigned to one structural component")]
    AmbiguousReferencePoint,
    #[error(
        "solid elements touch only through a node or edge; an explicit contact or tie is required"
    )]
    AmbiguousStructuralConnectivity,
    #[error("nullspace evidence is malformed or has been altered")]
    InvalidEvidence,
    #[error("nullspace evidence does not match the operator or constraints")]
    BindingMismatch,
    #[error("nullspace evidence was requested for an incompatible operator")]
    IncompatibleOperator,
    #[error("the physical operator does not contain the declared null mode")]
    NullModeViolation,
    #[error("a declared positive boundary anchor is absent from the operator")]
    MissingPositiveAnchor,
    #[error("nullspace evidence identity encoding failed: {0}")]
    Encoding(String),
}

pub(crate) struct ConnectorAnchoredStructuralFrameInput<'a> {
    pub structural: &'a runmat_analysis_core::StructuralModel,
    pub layout: &'a StructuralDofLayout,
    pub element_operator: &'a RealSparseMatrix,
    pub connector_operator: &'a RealSparseMatrix,
    pub combined_operator: &'a RealSparseMatrix,
    pub plan: &'a CanonicalStructuralConstraintPlan,
    pub constraints: &'a ResolvedConstraintMatrix,
    pub constrained_dofs: &'a [bool],
}

pub(crate) struct ConnectorAnchoredSolidInput<'a> {
    pub mesh: &'a SolverMeshArtifact,
    pub layout: &'a StructuralDofLayout,
    pub element_operator: &'a RealSparseMatrix,
    pub connector_operator: &'a RealSparseMatrix,
    pub combined_operator: &'a RealSparseMatrix,
    pub plan: &'a CanonicalStructuralConstraintPlan,
    pub constraints: &'a ResolvedConstraintMatrix,
    pub constrained_dofs: &'a [bool],
}

impl NullspaceEliminationEvidence {
    /// Prove elimination of one scalar constant mode per connected component.
    /// Used by thermal and electrical gradient forms; coefficients/provenance remain
    /// owned by their physics rather than by this shared mathematical proof.
    pub fn prove_scalar_diffusion(
        projection: &AnalysisProjection,
        operator: &RealSparseMatrix,
        prescribed: &[bool],
        positive_robin_weight: &[f64],
    ) -> Result<Self, NullspaceEvidenceError> {
        scalar::prove(projection, operator, prescribed, positive_robin_weight)
    }

    pub fn prove_solid(
        mesh: &SolverMeshArtifact,
        layout: &StructuralDofLayout,
        plan: &CanonicalStructuralConstraintPlan,
        constraints: &ResolvedConstraintMatrix,
        operator: &RealSparseMatrix,
        constrained_dofs: &[bool],
    ) -> Result<Self, NullspaceEvidenceError> {
        solid::prove(mesh, layout, plan, constraints, operator, constrained_dofs)
    }

    pub(crate) fn prove_connector_anchored_solid(
        input: ConnectorAnchoredSolidInput<'_>,
    ) -> Result<Self, NullspaceEvidenceError> {
        solid::prove_with_connector_anchors(input)
    }

    pub fn prove_structural_frame(
        structural: &runmat_analysis_core::StructuralModel,
        layout: &StructuralDofLayout,
        operator: &RealSparseMatrix,
        plan: &CanonicalStructuralConstraintPlan,
        constraints: &ResolvedConstraintMatrix,
        constrained_dofs: &[bool],
    ) -> Result<Self, NullspaceEvidenceError> {
        structural_frame::prove(
            structural,
            layout,
            operator,
            plan,
            constraints,
            constrained_dofs,
        )
    }

    pub(crate) fn prove_connector_anchored_structural_frame(
        input: ConnectorAnchoredStructuralFrameInput<'_>,
    ) -> Result<Self, NullspaceEvidenceError> {
        structural_frame::prove_with_connector_anchors(input)
    }

    pub(crate) fn prove_mixed_structural(
        input: MixedStructuralNullspaceInput<'_>,
    ) -> Result<Self, NullspaceEvidenceError> {
        mixed_structural::prove(input)
    }

    pub(super) fn new(
        kind: NullspaceKind,
        topology_identity: Digest,
        operator: &RealSparseMatrix,
        constraint_identity: Digest,
        constrained_dofs: &[bool],
        components: Vec<NullspaceComponentEvidence>,
    ) -> Result<Self, NullspaceEvidenceError> {
        let mut evidence = Self {
            schema_version: 1,
            kind,
            topology_identity,
            operator_pattern_identity: operator
                .pattern_identity()
                .map_err(|error| NullspaceEvidenceError::Encoding(error.to_string()))?,
            operator_values_identity: operator
                .values_identity()
                .map_err(|error| NullspaceEvidenceError::Encoding(error.to_string()))?,
            constraint_identity,
            constrained_dofs_identity: digest(CONSTRAINT_DOMAIN, &constrained_dofs)?,
            components,
            proof_identity: Digest::sha256([]),
        };
        evidence.proof_identity = evidence.expected_identity()?;
        evidence.validate()?;
        Ok(evidence)
    }

    pub(super) fn validate_for_system(
        &self,
        operator: &RealSparseMatrix,
        constrained_dofs: &[bool],
    ) -> Result<(), NullspaceEvidenceError> {
        self.validate()?;
        let pattern = operator
            .pattern_identity()
            .map_err(|error| NullspaceEvidenceError::Encoding(error.to_string()))?;
        let values = operator
            .values_identity()
            .map_err(|error| NullspaceEvidenceError::Encoding(error.to_string()))?;
        if self.operator_pattern_identity != pattern
            || self.operator_values_identity != values
            || self.constrained_dofs_identity != digest(CONSTRAINT_DOMAIN, &constrained_dofs)?
        {
            return Err(NullspaceEvidenceError::BindingMismatch);
        }
        Ok(())
    }

    pub(super) fn bind_prepared(
        &self,
        matrix: &RealSparseMatrix,
        rhs: &[f64],
    ) -> Result<PreparedNullspaceEvidence, NullspaceEvidenceError> {
        self.validate()?;
        let mut binding = PreparedNullspaceEvidence {
            source: self.clone(),
            prepared_pattern_identity: matrix
                .pattern_identity()
                .map_err(|error| NullspaceEvidenceError::Encoding(error.to_string()))?,
            prepared_values_identity: matrix
                .values_identity()
                .map_err(|error| NullspaceEvidenceError::Encoding(error.to_string()))?,
            prepared_rhs_identity: digest(RHS_DOMAIN, &rhs)?,
            binding_identity: Digest::sha256([]),
        };
        binding.binding_identity = binding.expected_identity()?;
        binding.validate()?;
        Ok(binding)
    }

    fn validate(&self) -> Result<(), NullspaceEvidenceError> {
        if self.schema_version != 1
            || self.components.is_empty()
            || self
                .components
                .iter()
                .enumerate()
                .any(|(index, component)| {
                    component.component_index != index
                        || component.node_identities.is_empty()
                        || !component
                            .node_identities
                            .windows(2)
                            .all(|pair| pair[0] < pair[1])
                        || component.null_mode_count == 0
                        || component.projected_anchor_rank != component.null_mode_count
                        || !anchoring_matches(component, self.kind)
                })
            || self.expected_identity()? != self.proof_identity
        {
            return Err(NullspaceEvidenceError::InvalidEvidence);
        }
        let unique_nodes = self
            .components
            .iter()
            .flat_map(|component| component.node_identities.iter().copied())
            .collect::<BTreeSet<_>>();
        if unique_nodes.len()
            != self
                .components
                .iter()
                .map(|component| component.node_identities.len())
                .sum::<usize>()
        {
            return Err(NullspaceEvidenceError::InvalidEvidence);
        }
        Ok(())
    }

    fn expected_identity(&self) -> Result<Digest, NullspaceEvidenceError> {
        digest(
            EVIDENCE_DOMAIN,
            &(
                self.schema_version,
                self.kind,
                self.topology_identity,
                self.operator_pattern_identity,
                self.operator_values_identity,
                self.constraint_identity,
                self.constrained_dofs_identity,
                &self.components,
            ),
        )
    }
}

pub(crate) struct MixedStructuralNullspaceInput<'a> {
    pub mesh: &'a SolverMeshArtifact,
    pub structural: &'a runmat_analysis_core::StructuralModel,
    pub topology:
        &'a crate::physics::structural::topology::node_space::CanonicalStructuralNodeSpace,
    pub layout: &'a StructuralDofLayout,
    pub element_operator: &'a RealSparseMatrix,
    pub connector_operator: Option<&'a RealSparseMatrix>,
    pub combined_operator: &'a RealSparseMatrix,
    pub constraints: &'a ResolvedConstraintMatrix,
    pub constraint_plan: &'a crate::constraints::CanonicalStructuralConstraintPlan,
    pub constrained_dofs: &'a [bool],
}

impl PreparedNullspaceEvidence {
    pub(super) const fn identity(&self) -> Digest {
        self.binding_identity
    }

    fn validate(&self) -> Result<(), NullspaceEvidenceError> {
        self.source.validate()?;
        if self.binding_identity != self.expected_identity()? {
            return Err(NullspaceEvidenceError::InvalidEvidence);
        }
        Ok(())
    }

    fn expected_identity(&self) -> Result<Digest, NullspaceEvidenceError> {
        digest(
            BINDING_DOMAIN,
            &(
                self.source.proof_identity,
                self.prepared_pattern_identity,
                self.prepared_values_identity,
                self.prepared_rhs_identity,
            ),
        )
    }
}

fn anchoring_matches(component: &NullspaceComponentEvidence, kind: NullspaceKind) -> bool {
    match (&component.anchoring, kind) {
        (
            NullspaceAnchoring::PrescribedDof { count },
            NullspaceKind::ScalarConstantPerComponent,
        )
        | (
            NullspaceAnchoring::PositiveRobin { count },
            NullspaceKind::ScalarConstantPerComponent,
        ) => *count > 0,
        (
            NullspaceAnchoring::PrescribedAndPositiveRobin {
                prescribed_count,
                robin_count,
            },
            NullspaceKind::ScalarConstantPerComponent,
        ) => *prescribed_count > 0 && *robin_count > 0,
        (
            NullspaceAnchoring::ConstraintRowSpace,
            NullspaceKind::SolidRigidBody3d
            | NullspaceKind::StructuralFrameRigidBody3d
            | NullspaceKind::MixedStructuralRigidBody3d,
        ) => true,
        (
            NullspaceAnchoring::OperatorAndConstraintRowSpace,
            NullspaceKind::SolidRigidBody3d
            | NullspaceKind::StructuralFrameRigidBody3d
            | NullspaceKind::MixedStructuralRigidBody3d,
        ) => true,
        _ => false,
    }
}

pub(super) fn numerical_rank(mut rows: Vec<Vec<f64>>) -> usize {
    for row in &mut rows {
        let scale = row.iter().map(|value| value.abs()).fold(0.0_f64, f64::max);
        if scale > 0.0 && scale.is_finite() {
            for value in row {
                *value /= scale;
            }
        }
    }
    let columns = rows.first().map(Vec::len).unwrap_or(0);
    let scale = rows
        .iter()
        .flatten()
        .map(|value| value.abs())
        .fold(0.0_f64, f64::max)
        .max(1.0);
    let tolerance = scale * RANK_TOLERANCE;
    let mut rank = 0;
    for column in 0..columns {
        let pivot = (rank..rows.len()).max_by(|left, right| {
            rows[*left][column]
                .abs()
                .total_cmp(&rows[*right][column].abs())
        });
        let Some(pivot) = pivot.filter(|pivot| rows[*pivot][column].abs() > tolerance) else {
            continue;
        };
        rows.swap(rank, pivot);
        let divisor = rows[rank][column];
        for value in &mut rows[rank][column..] {
            *value /= divisor;
        }
        for row in 0..rows.len() {
            if row == rank {
                continue;
            }
            let factor = rows[row][column];
            if factor.abs() <= tolerance {
                continue;
            }
            for entry in column..columns {
                rows[row][entry] -= factor * rows[rank][entry];
            }
        }
        rank += 1;
        if rank == rows.len() {
            break;
        }
    }
    rank
}

pub(super) fn validate_operator(
    operator: &RealSparseMatrix,
    role: OperatorRole,
    definiteness: MatrixDefiniteness,
) -> Result<(), NullspaceEvidenceError> {
    if operator.role() != role
        || operator.properties().symmetry != MatrixSymmetry::Symmetric
        || operator.properties().definiteness != definiteness
    {
        return Err(NullspaceEvidenceError::IncompatibleOperator);
    }
    Ok(())
}

pub(super) fn verify_null_mode(
    operator: &RealSparseMatrix,
    mode: &[f64],
    applied: &[f64],
) -> Result<(), NullspaceEvidenceError> {
    let mode_scale = mode.iter().map(|value| value.abs()).fold(0.0_f64, f64::max);
    let residual = applied
        .iter()
        .map(|value| value.abs())
        .fold(0.0_f64, f64::max);
    let reference = (operator_scale(operator) * mode_scale).max(1.0);
    if mode_scale == 0.0
        || !residual.is_finite()
        || residual > reference * NULL_MODE_RELATIVE_TOLERANCE
    {
        return Err(NullspaceEvidenceError::NullModeViolation);
    }
    Ok(())
}

pub(super) fn operator_scale(operator: &RealSparseMatrix) -> f64 {
    operator
        .pattern()
        .row_offsets()
        .windows(2)
        .map(|row| {
            operator.values()[row[0] as usize..row[1] as usize]
                .iter()
                .map(|value| value.abs())
                .sum::<f64>()
        })
        .fold(0.0_f64, f64::max)
}

pub(super) fn digest<T: Serialize>(
    domain: &'static str,
    value: &T,
) -> Result<Digest, NullspaceEvidenceError> {
    // Streamed: evidence over mesh-sized bases is hashed without materializing its encoding.
    crate::contracts::canonical_digest::canonical_contract_digest(
        EVIDENCE_PREFIX,
        domain,
        value,
        LIMITS,
    )
    .map_err(|error| NullspaceEvidenceError::Encoding(error.to_string()))
}
