//! Physical small-strain solid residual/tangent with immutable committed point history.
//! No global dense matrix, saturation proxy, first-material selection or trial-state mutation.
use std::collections::{BTreeMap, BTreeSet};

use runmat_analysis_core::{AnalysisModel, AnalysisProcedure, StructuralKinematics};
use runmat_canonical_codec::{encode_contract, CanonicalLimits};
use runmat_execution::Digest;
use runmat_meshing_core::{ElementOrder, SolverMeshArtifact, StableDigest};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::{dofs::StructuralDofKind, AssemblySummary};
use crate::{
    algebra::{
        ContributionDomain, ContributionId, ContributionProvenance, MatrixProperties, OperatorRole,
        RealSparseAssembler, RealSparseMatrix, SparsePattern,
    },
    physics::structural::{
        plasticity::{IsotropicConstitutiveModel, PlasticityPointState},
        topology::node_space::CanonicalStructuralNodeSpace,
    },
    properties::{compile_structural_property_map, CompiledPropertyMapIdentity},
    topology::{AnalysisProjection, CanonicalSolidIntegrationPlan, CanonicalSolidIntegrationRule},
};

mod authority;
mod contact_layout;
mod evaluation;
mod follower;
pub(crate) use follower::FollowerLoadOperator;
mod history;
mod preparation;
pub(crate) mod resources;
#[cfg(test)]
pub(crate) mod tests;

#[derive(Debug, Error)]
pub(crate) enum NonlinearSolidError {
    #[error("nonlinear solid numerical accounting failed: {0}")]
    Accounting(runmat_simulation_usage::LedgerError),
    #[error("nonlinear solid operation cancelled")]
    Cancelled,
    #[error("unsupported nonlinear solid envelope: {0}")]
    Unsupported(String),
    #[error("invalid nonlinear solid authority: {0}")]
    Authority(String),
    #[error("invalid nonlinear solid history: {0}")]
    History(String),
    #[error("nonlinear constitutive update failed: {0}")]
    Constitutive(String),
    #[error("nonlinear sparse tangent failed: {0}")]
    Tangent(String),
}

impl From<crate::topology::AnalysisProjectionError> for NonlinearSolidError {
    fn from(error: crate::topology::AnalysisProjectionError) -> Self {
        match error {
            crate::topology::AnalysisProjectionError::Cancelled => Self::Cancelled,
            crate::topology::AnalysisProjectionError::Accounting(error) => Self::Accounting(error),
            other => Self::Authority(other.to_string()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct NonlinearSolidPointIdentity {
    pub element_identity: StableDigest,
    pub quadrature_ordinal: usize,
}

/// Exact ordered integration-point history, bound to mesh, compiled properties and DOF basis.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct NonlinearSolidState {
    pub basis_identity: Digest,
    pub integration_identity: Digest,
    pub mesh_digest: StableDigest,
    pub property_identity: CompiledPropertyMapIdentity,
    pub point_identities: Vec<NonlinearSolidPointIdentity>,
    pub points: Vec<PlasticityPointState>,
}

#[derive(Debug, Clone)]
pub(crate) struct NonlinearSolidPointEvidence {
    pub identity: NonlinearSolidPointIdentity,
    pub position_m: [f64; 3],
    pub volume_weight_m3: f64,
    pub total_engineering_strain: [f64; 6],
    pub stress_pa: [f64; 6],
    pub elastic_energy_density_pa: f64,
    pub hardening_energy_density_pa: f64,
    pub plastic_dissipation_increment_pa: f64,
}

#[derive(Debug, Clone)]
pub(crate) struct NonlinearSolidElementEvidence {
    pub element_identity: StableDigest,
    pub element_id: u64,
    pub volume_m3: f64,
    pub mean_total_engineering_strain: [f64; 6],
    pub mean_stress_pa: [f64; 6],
    pub elastic_energy_j: runmat_types::physical::Energy,
    pub hardening_energy_j: runmat_types::physical::Energy,
    pub plastic_dissipation_increment_j: f64,
}

pub(crate) struct NonlinearSolidEvaluation {
    pub internal_force: Vec<f64>,
    pub tangent: RealSparseMatrix,
    /// A trial, not accepted history. Caller commits only after equilibrium convergence.
    pub trial_state: NonlinearSolidState,
    pub point_evidence: Vec<NonlinearSolidPointEvidence>,
    pub element_evidence: Vec<NonlinearSolidElementEvidence>,
}

#[derive(Serialize)]
struct PreparedPoint {
    identity: NonlinearSolidPointIdentity,
    position_m: [f64; 3],
    weight_m3: f64,
    gradients: Vec<[f64; 3]>,
}

struct PreparedBodyNode {
    identity: StableDigest,
    coordinates_m:
        runmat_types::physical::Quantity<runmat_types::physical::kind::Position, [f64; 3]>,
    rows: [usize; 3],
}

struct PreparedElement {
    identity: StableDigest,
    element_id: u64,
    dofs: Vec<[usize; 3]>,
    constitutive: IsotropicConstitutiveModel,
    points: Vec<PreparedPoint>,
}

pub(crate) struct CanonicalNonlinearSolidBasis {
    identity: Digest,
    kinematics: StructuralKinematics,
    integration_identity: Digest,
    mesh_digest: StableDigest,
    property_identity: CompiledPropertyMapIdentity,
    dof_count: usize,
    tangent_pattern: SparsePattern,
    nodes: Vec<PreparedBodyNode>,
    elements: Vec<PreparedElement>,
}

impl CanonicalNonlinearSolidBasis {
    pub(crate) fn identity(&self) -> Digest {
        self.identity
    }
    pub(crate) fn dof_count(&self) -> usize {
        self.dof_count
    }
    pub(crate) fn point_count(&self) -> usize {
        self.elements.iter().map(|e| e.points.len()).sum()
    }

    pub(crate) fn zero_state(&self) -> NonlinearSolidState {
        NonlinearSolidState {
            basis_identity: self.identity,
            integration_identity: self.integration_identity,
            mesh_digest: self.mesh_digest,
            property_identity: self.property_identity.clone(),
            point_identities: self
                .elements
                .iter()
                .flat_map(|e| e.points.iter().map(|p| p.identity.clone()))
                .collect(),
            points: vec![PlasticityPointState::default(); self.point_count()],
        }
    }

    pub(crate) fn validate_state(
        &self,
        state: &NonlinearSolidState,
    ) -> Result<(), NonlinearSolidError> {
        if state.basis_identity != self.identity
            || state.integration_identity != self.integration_identity
            || state.mesh_digest != self.mesh_digest
            || state.property_identity != self.property_identity
            || state.points.len() != self.point_count()
            || state.point_identities.len() != self.point_count()
        {
            return Err(NonlinearSolidError::History(
                "mesh/property/basis or point inventory differs".into(),
            ));
        }
        let mut ordinal = 0;
        for element in &self.elements {
            for point in &element.points {
                if state.point_identities[ordinal] != point.identity {
                    return Err(NonlinearSolidError::History(
                        "ordered element/quadrature identity differs".into(),
                    ));
                }
                element
                    .constitutive
                    .update([0.0; 6], &state.points[ordinal])
                    .map_err(|e| NonlinearSolidError::History(e.to_string()))?;
                ordinal += 1;
            }
        }
        Ok(())
    }
}
