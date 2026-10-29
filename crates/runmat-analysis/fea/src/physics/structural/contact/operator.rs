//! Prepared global contact operator. Configuration trials are not accepted body states.
use super::*;
use crate::{
    algebra::{RealSparseMatrix, SparsePattern, SparsePatternError},
    topology::{
        AnalysisProjection, BoundaryPairingError, BoundaryPairingPolicy, BoundaryStationaryPolicy,
        BoundarySurfaceIndex,
    },
};
use runmat_analysis_core::{AnalysisInterface, AnalysisInterfaceKind};
use runmat_execution::Digest;
use serde::{Deserialize, Serialize};
use thiserror::Error;
mod admission;
mod evaluation;
mod output;
mod preparation;
pub use admission::ContactPreparationDemand;
mod resources;
pub use resources::ContactOperatorResources;
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ContactInterfacePolicy {
    pub penalty: ContactPenaltyPolicy,
    pub pairing: BoundaryPairingPolicy,
    pub stationary: BoundaryStationaryPolicy,
    pub maximum_initial_penetration_m: f64,
}
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ContactOperatorPolicy {
    pub maximum_dofs: usize,
    pub maximum_reference_points: usize,
    pub maximum_face_pairs: usize,
    pub maximum_coordinates: usize,
    pub maximum_patches_per_trial: u64,
    pub maximum_refinement_evaluations_per_trial: u64,
    pub maximum_preparation_resident_bytes: u64,
    pub maximum_preparation_scalar_operations: u64,
}
impl ContactOperatorPolicy {
    /// The identity source grows with the contact support (node rows, interfaces); its
    /// construction bytes are charged to `maximum_preparation_resident_bytes`, so the codec
    /// bounds it only by the wire range.
    fn identity_limits(self) -> runmat_canonical_codec::CanonicalLimits {
        crate::contracts::canonical_digest::wire_bounded(
            runmat_canonical_codec::CanonicalLimits::new(0, 0, 65536, 32),
        )
    }
}
#[derive(Debug, Error)]
pub enum ContactOperatorError {
    #[error("invalid contact operator source, controls or original state")]
    Invalid,
    #[error("contact operator exceeds its declared envelope")]
    Budget,
    #[error("contact feature transition requires qualified history transfer")]
    FeatureTransition,
    #[error("initial contact penetration exceeds its declared limit")]
    InitialPenetration,
    #[error("contact operator cancelled")]
    Cancelled,
    #[error(transparent)]
    History(#[from] ContactHistoryError),
    #[error(transparent)]
    Selection(#[from] ContactSurfaceError),
    #[error(transparent)]
    Pairing(#[from] BoundaryPairingError),
    #[error(transparent)]
    Support(#[from] ContactSupportError),
    #[error(transparent)]
    Sparse(#[from] SparsePatternError),
    #[error(transparent)]
    Matrix(#[from] crate::algebra::RealSparseMatrixError),
}
struct PreparedInterface<'a> {
    inventory: ContactReferenceInventory<'a>,
    points: Vec<ContactPreparedPoint<'a>>,
    policy: ContactInterfacePolicy,
}
pub struct CanonicalContactOperator<'a> {
    projection: &'a AnalysisProjection,
    node_rows: &'a [[usize; 3]],
    authored_interfaces: &'a [AnalysisInterface],
    interfaces: Vec<PreparedInterface<'a>>,
    pattern: SparsePattern,
    identity: Digest,
    execution_identity: Digest,
    policy: ContactOperatorPolicy,
    point_count: usize,
    identity_construction_bytes: u64,
    construction_scratch_bytes: u64,
    preparation_demand: ContactPreparationDemand,
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ContactAssemblyDiagnostics {
    pub visited_patches: u64,
    pub tested_faces: u64,
    /// Conservative geometry-evaluation bound from actual steps/backtracks, not measured work.
    pub refinement_evaluations_bound: u64,
    pub active_points: usize,
    pub open_points: usize,
    pub stick_points: usize,
    pub slip_points: usize,
    pub maximum_penetration_m: f64,
    pub normal_elastic_energy_j: f64,
    pub tangential_elastic_energy_j: f64,
    pub frictional_dissipation_increment_j: f64,
    pub algorithmic_dissipation_increment_j: f64,
    pub separation_release_energy_j: f64,
}
pub struct ContactAssemblyTrial {
    pub residual: Vec<f64>,
    pub tangent: RealSparseMatrix,
    pub configuration: ContactConfigurationState,
    pub diagnostics: ContactAssemblyDiagnostics,
}
impl CanonicalContactOperator<'_> {
    /// Compare consumed source content, not only provenance digests. Mutable
    /// projection fields cannot become trusted geometry through a retained hash.
    pub(crate) fn matches_sources(
        &self,
        projection: &AnalysisProjection,
        interfaces: &[AnalysisInterface],
    ) -> bool {
        self.projection == projection && self.authored_interfaces == interfaces
    }
    pub fn preparation_demand(&self) -> ContactPreparationDemand {
        self.preparation_demand
    }
    pub fn identity(&self) -> Digest {
        self.identity
    }
    pub fn execution_identity(&self) -> Digest {
        self.execution_identity
    }
    pub(crate) fn projection(&self) -> &AnalysisProjection {
        self.projection
    }
    pub(crate) fn node_rows(&self) -> &[[usize; 3]] {
        self.node_rows
    }
    pub fn point_count(&self) -> usize {
        self.point_count
    }
    pub fn pattern(&self) -> &SparsePattern {
        &self.pattern
    }
    fn check_positions(&self, positions: &[[f64; 3]]) -> Result<(), ContactOperatorError> {
        if crate::is_fea_cancelled() {
            return Err(ContactOperatorError::Cancelled);
        }
        if positions.len() != self.projection.nodes.len()
            || positions.iter().flatten().any(|x| !x.is_finite())
        {
            return Err(ContactOperatorError::Invalid);
        }
        Ok(())
    }
    fn check_state_header(
        &self,
        state: &ContactConfigurationState,
        expected_ordinal: u64,
    ) -> Result<(), ContactOperatorError> {
        if state.schema != 1
            || state.scope != self.identity
            || state.ordinal != expected_ordinal
            || state.points.len() != self.point_count
        {
            return Err(ContactOperatorError::Invalid);
        }
        self.check_positions(&state.positions_m)
    }
    pub fn validate_configuration(
        &self,
        state: &ContactConfigurationState,
        expected_ordinal: u64,
    ) -> Result<(), ContactOperatorError> {
        self.check_state_header(state, expected_ordinal)?;
        let mut ordinal = 0;
        for interface in &self.interfaces {
            for point in &interface.points {
                let binding = point.bind(&interface.inventory)?;
                binding.material_history(
                    &state.points[ordinal],
                    expected_ordinal,
                    &state.positions_m,
                )?;
                ordinal += 1;
            }
        }
        if crate::is_fea_cancelled() {
            return Err(ContactOperatorError::Cancelled);
        }
        Ok(())
    }
    fn index(
        &self,
        interface: &PreparedInterface<'_>,
        positions: &[[f64; 3]],
    ) -> Result<BoundarySurfaceIndex, ContactOperatorError> {
        Ok(BoundarySurfaceIndex::derive(
            self.projection,
            interface.inventory.primary_face_indices(),
            positions,
            interface.policy.pairing.maximum_faces,
        )?)
    }
}
fn reserve<T>(count: usize) -> Result<Vec<T>, ContactOperatorError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .map_err(|_| ContactOperatorError::Budget)?;
    Ok(values)
}
#[cfg(test)]
pub(crate) mod tests;
