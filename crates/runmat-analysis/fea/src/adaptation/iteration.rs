//! Canonical solver-owned evidence and convergence decisions for adaptive structural iterations.

use runmat_meshing_core::{
    CanonicalMeshingContract, SolverMeshAdaptationLineage, SolverMeshArtifact,
    SolverMeshTransferMap, StableDigest,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};

use super::{SolverFieldTransferErrorEvidence, StructuralRecoveryEstimate};

mod decision;
mod validation;

use decision::{decide, validate_policy};
use validation::{
    validate_estimator_against, validate_solver_result, validate_target_quantity,
    validate_transfer_errors, validate_transfer_errors_against,
};

const CODEC_PREFIX: &[u8] = b"runmat-analysis-fea-canonical-cbor/v1\0";
const CODEC_DOMAIN: &str = "analysis.fea.structural-adaptation-iteration/v1";
const CODEC_LIMITS: CanonicalLimits = CanonicalLimits::new(64 * 1024 * 1024, 1_000_000, 4096, 32);
pub const STRUCTURAL_ADAPTATION_ITERATION_SCHEMA_VERSION: u16 = 1;

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuralAdaptationPolicy {
    pub estimator_tolerance: f64,
    pub minimum_estimator_reduction: f64,
    pub target_absolute_tolerance: f64,
    pub target_relative_tolerance: f64,
    pub maximum_transfer_relative_error: f64,
}

impl Default for StructuralAdaptationPolicy {
    fn default() -> Self {
        Self {
            estimator_tolerance: 1.0e-3,
            minimum_estimator_reduction: 0.05,
            target_absolute_tolerance: 1.0e-6,
            target_relative_tolerance: 1.0e-3,
            maximum_transfer_relative_error: 0.05,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuralAdaptationSolverResult {
    pub result_digest: StableDigest,
    pub converged: bool,
    pub iteration_count: u64,
    pub normalized_residual: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuralTargetQuantity {
    pub quantity_id: String,
    pub value: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StructuralAdaptationDecisionStatus {
    Continue,
    Converged,
    Rejected,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuralAdaptationConvergenceDecision {
    pub status: StructuralAdaptationDecisionStatus,
    pub estimator_reduction: Option<f64>,
    pub target_absolute_change: Option<f64>,
    pub target_relative_change: Option<f64>,
    pub solver_converged: bool,
    pub transfer_accepted: bool,
    pub estimator_reduction_accepted: bool,
    pub estimator_target_met: bool,
    pub target_quantity_converged: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuralAdaptationIteration {
    pub schema_version: u16,
    pub iteration_index: u64,
    pub previous_iteration_digest: Option<StableDigest>,
    pub source_solver_artifact_digest: StableDigest,
    pub target_solver_artifact_digest: StableDigest,
    pub adaptation_lineage_digest: StableDigest,
    pub transfer_map_digest: StableDigest,
    pub estimator: StructuralRecoveryEstimate,
    pub transfer_errors: Vec<SolverFieldTransferErrorEvidence>,
    pub solver_result: StructuralAdaptationSolverResult,
    pub target_quantity: StructuralTargetQuantity,
    pub previous_estimator_error: Option<f64>,
    pub previous_target_quantity_value: Option<f64>,
    pub policy: StructuralAdaptationPolicy,
    pub decision: StructuralAdaptationConvergenceDecision,
}

#[derive(Clone, Copy)]
pub struct StructuralAdaptationIterationInput<'a> {
    pub source_artifact: &'a SolverMeshArtifact,
    pub target_artifact: &'a SolverMeshArtifact,
    pub transfer_map: &'a SolverMeshTransferMap,
    pub lineage: &'a SolverMeshAdaptationLineage,
    pub estimator: &'a StructuralRecoveryEstimate,
    pub transfer_errors: &'a [SolverFieldTransferErrorEvidence],
    pub solver_result: &'a StructuralAdaptationSolverResult,
    pub target_quantity: &'a StructuralTargetQuantity,
    pub previous: Option<&'a StructuralAdaptationIteration>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StructuralAdaptationIterationError {
    InvalidPolicy,
    InvalidArtifact(String),
    InvalidLineage(String),
    InvalidEstimator,
    InvalidTransferEvidence,
    InvalidSolverResult,
    InvalidTargetQuantity,
    InvalidIterationChain,
    InvalidDecision,
    Codec(String),
}

pub fn build_structural_adaptation_iteration(
    input: StructuralAdaptationIterationInput<'_>,
    policy: StructuralAdaptationPolicy,
) -> Result<StructuralAdaptationIteration, StructuralAdaptationIterationError> {
    validate_policy(policy)?;
    input.source_artifact.validate().map_err(|failure| {
        StructuralAdaptationIterationError::InvalidArtifact(failure.to_string())
    })?;
    input.target_artifact.validate().map_err(|failure| {
        StructuralAdaptationIterationError::InvalidArtifact(failure.to_string())
    })?;
    input
        .lineage
        .validate_against(
            input.source_artifact,
            input.target_artifact,
            input.transfer_map,
        )
        .map_err(|failure| {
            StructuralAdaptationIterationError::InvalidLineage(failure.to_string())
        })?;
    input
        .estimator
        .validate()
        .map_err(|_| StructuralAdaptationIterationError::InvalidEstimator)?;
    if input.estimator.solver_artifact_digest != input.target_artifact.canonical_digest {
        return Err(StructuralAdaptationIterationError::InvalidEstimator);
    }
    validate_estimator_against(input.estimator, input.target_artifact)?;
    validate_transfer_errors(
        input.transfer_errors,
        input.source_artifact.canonical_digest,
        input.target_artifact.canonical_digest,
    )?;
    validate_transfer_errors_against(
        input.transfer_errors,
        input.source_artifact,
        input.target_artifact,
    )?;
    validate_solver_result(input.solver_result)?;
    validate_target_quantity(input.target_quantity)?;
    let previous = previous_evidence(
        input.previous,
        input.source_artifact.canonical_digest,
        &input.target_quantity.quantity_id,
    )?;
    let decision = decide(
        input.estimator.total_error,
        input.transfer_errors,
        input.solver_result,
        input.target_quantity.value,
        previous.previous_estimator_error,
        previous.previous_target_quantity_value,
        policy,
    );
    let iteration = StructuralAdaptationIteration {
        schema_version: STRUCTURAL_ADAPTATION_ITERATION_SCHEMA_VERSION,
        iteration_index: previous.iteration_index,
        previous_iteration_digest: previous.previous_iteration_digest,
        source_solver_artifact_digest: input.source_artifact.canonical_digest,
        target_solver_artifact_digest: input.target_artifact.canonical_digest,
        adaptation_lineage_digest: input.lineage.canonical_digest().map_err(|failure| {
            StructuralAdaptationIterationError::InvalidLineage(failure.to_string())
        })?,
        transfer_map_digest: input.transfer_map.canonical_digest().map_err(|failure| {
            StructuralAdaptationIterationError::InvalidLineage(failure.to_string())
        })?,
        estimator: input.estimator.clone(),
        transfer_errors: input.transfer_errors.to_vec(),
        solver_result: input.solver_result.clone(),
        target_quantity: input.target_quantity.clone(),
        previous_estimator_error: previous.previous_estimator_error,
        previous_target_quantity_value: previous.previous_target_quantity_value,
        policy,
        decision,
    };
    iteration.validate()?;
    Ok(iteration)
}

impl StructuralAdaptationIteration {
    pub fn validate(&self) -> Result<(), StructuralAdaptationIterationError> {
        if self.schema_version != STRUCTURAL_ADAPTATION_ITERATION_SCHEMA_VERSION
            || self.source_solver_artifact_digest == StableDigest::ZERO
            || self.target_solver_artifact_digest == StableDigest::ZERO
            || self.source_solver_artifact_digest == self.target_solver_artifact_digest
            || self.adaptation_lineage_digest == StableDigest::ZERO
            || self.transfer_map_digest == StableDigest::ZERO
            || self.estimator.solver_artifact_digest != self.target_solver_artifact_digest
        {
            return Err(StructuralAdaptationIterationError::InvalidIterationChain);
        }
        validate_policy(self.policy)?;
        self.estimator
            .validate()
            .map_err(|_| StructuralAdaptationIterationError::InvalidEstimator)?;
        validate_transfer_errors(
            &self.transfer_errors,
            self.source_solver_artifact_digest,
            self.target_solver_artifact_digest,
        )?;
        validate_solver_result(&self.solver_result)?;
        validate_target_quantity(&self.target_quantity)?;
        let has_previous = self.previous_iteration_digest.is_some();
        if has_previous != (self.iteration_index > 0)
            || has_previous != self.previous_estimator_error.is_some()
            || has_previous != self.previous_target_quantity_value.is_some()
            || self.previous_iteration_digest == Some(StableDigest::ZERO)
        {
            return Err(StructuralAdaptationIterationError::InvalidIterationChain);
        }
        let expected = decide(
            self.estimator.total_error,
            &self.transfer_errors,
            &self.solver_result,
            self.target_quantity.value,
            self.previous_estimator_error,
            self.previous_target_quantity_value,
            self.policy,
        );
        if self.decision != expected {
            return Err(StructuralAdaptationIterationError::InvalidDecision);
        }
        Ok(())
    }

    pub fn validate_against(
        &self,
        source: &SolverMeshArtifact,
        target: &SolverMeshArtifact,
        transfer: &SolverMeshTransferMap,
        lineage: &SolverMeshAdaptationLineage,
    ) -> Result<(), StructuralAdaptationIterationError> {
        self.validate()?;
        lineage
            .validate_against(source, target, transfer)
            .map_err(|failure| {
                StructuralAdaptationIterationError::InvalidLineage(failure.to_string())
            })?;
        validate_estimator_against(&self.estimator, target)?;
        validate_transfer_errors_against(&self.transfer_errors, source, target)?;
        if self.source_solver_artifact_digest != source.canonical_digest
            || self.target_solver_artifact_digest != target.canonical_digest
            || self.adaptation_lineage_digest
                != lineage.canonical_digest().map_err(|failure| {
                    StructuralAdaptationIterationError::InvalidLineage(failure.to_string())
                })?
            || self.transfer_map_digest
                != transfer.canonical_digest().map_err(|failure| {
                    StructuralAdaptationIterationError::InvalidLineage(failure.to_string())
                })?
        {
            return Err(StructuralAdaptationIterationError::InvalidIterationChain);
        }
        Ok(())
    }

    pub fn canonical_encode(&self) -> Result<Vec<u8>, StructuralAdaptationIterationError> {
        self.validate()?;
        runmat_canonical_codec::encode_contract(CODEC_PREFIX, CODEC_DOMAIN, self, CODEC_LIMITS)
            .map_err(codec_error)
    }

    pub fn canonical_decode(bytes: &[u8]) -> Result<Self, StructuralAdaptationIterationError> {
        let record: Self = runmat_canonical_codec::decode_contract(
            CODEC_PREFIX,
            CODEC_DOMAIN,
            bytes,
            CODEC_LIMITS,
        )
        .map_err(codec_error)?;
        record.validate()?;
        Ok(record)
    }

    pub fn canonical_digest(&self) -> Result<StableDigest, StructuralAdaptationIterationError> {
        Ok(StableDigest::from_bytes(
            Sha256::digest(self.canonical_encode()?).into(),
        ))
    }
}

struct PreviousIterationEvidence {
    iteration_index: u64,
    previous_iteration_digest: Option<StableDigest>,
    previous_estimator_error: Option<f64>,
    previous_target_quantity_value: Option<f64>,
}

fn previous_evidence(
    previous: Option<&StructuralAdaptationIteration>,
    source_digest: StableDigest,
    target_quantity_id: &str,
) -> Result<PreviousIterationEvidence, StructuralAdaptationIterationError> {
    let Some(previous) = previous else {
        return Ok(PreviousIterationEvidence {
            iteration_index: 0,
            previous_iteration_digest: None,
            previous_estimator_error: None,
            previous_target_quantity_value: None,
        });
    };
    previous.validate()?;
    if previous.target_solver_artifact_digest != source_digest
        || previous.target_quantity.quantity_id != target_quantity_id
        || previous.decision.status != StructuralAdaptationDecisionStatus::Continue
    {
        return Err(StructuralAdaptationIterationError::InvalidIterationChain);
    }
    let iteration_index = previous
        .iteration_index
        .checked_add(1)
        .ok_or(StructuralAdaptationIterationError::InvalidIterationChain)?;
    Ok(PreviousIterationEvidence {
        iteration_index,
        previous_iteration_digest: Some(previous.canonical_digest()?),
        previous_estimator_error: Some(previous.estimator.total_error),
        previous_target_quantity_value: Some(previous.target_quantity.value),
    })
}

fn codec_error(error: CanonicalCodecError) -> StructuralAdaptationIterationError {
    StructuralAdaptationIterationError::Codec(error.to_string())
}

#[cfg(test)]
mod tests;
use runmat_canonical_codec::{CanonicalCodecError, CanonicalLimits};
