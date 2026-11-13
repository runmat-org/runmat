//! Structural adaptation lineage with shared study accuracy as the sole acceptance authority.
use runmat_analysis_core::AccuracyPlan;
use runmat_canonical_codec::{CanonicalCodecError, CanonicalLimits};
use runmat_meshing_core::{
    CanonicalMeshingContract, SolverMeshAdaptationLineage, SolverMeshArtifact,
    SolverMeshTransferMap, StableDigest,
};
use sha2::{Digest as _, Sha256};
mod accuracy;
mod constraints;
mod contract;
mod decision;
mod validation;
use super::{SolverFieldTransferErrorEvidence, StructuralRecoveryEstimate};
pub use constraints::StructuralAdaptationConstraintLineage;
pub use contract::*;
use validation::*;

const CODEC_PREFIX: &[u8] = b"runmat-analysis-fea-canonical-cbor/v1\0";
const CODEC_DOMAIN: &str = "analysis.fea.structural-adaptation-iteration/v3";
const CODEC_LIMITS: CanonicalLimits = CanonicalLimits::platform_content(4096, 32);
pub const STRUCTURAL_ADAPTATION_ITERATION_SCHEMA_VERSION: u16 = 3;

pub fn build_structural_adaptation_iteration(
    input: StructuralAdaptationIterationInput<'_>,
    plan: &AccuracyPlan,
) -> Result<StructuralAdaptationIteration, StructuralAdaptationIterationError> {
    let constraints = StructuralAdaptationConstraintLineage::derive(
        input.model,
        input.source_artifact,
        input.target_artifact,
    )?;
    let index = input
        .previous
        .map(|p| p.iteration_index.checked_add(1))
        .unwrap_or(Some(0))
        .ok_or(StructuralAdaptationIterationError::InvalidIterationChain)?;
    let metrics = decision::changes(
        input.source_estimator.total_error,
        input.estimator.total_error,
        input.source_target_quantity.value,
        input.target_quantity.value,
    )?;
    let evaluation = accuracy::evaluate(accuracy::AccuracyInput {
        plan,
        source_estimator: input.source_estimator,
        estimator: input.estimator,
        transfer_errors: input.transfer_errors,
        solver_result: input.solver_result,
        target_quantity: input.target_quantity,
        absolute_change: metrics.1,
        relative_change: metrics.2,
    })?;
    let decision = decision::decide(
        plan,
        &evaluation,
        index,
        input.source_solver_result.converged && input.solver_result.converged,
        metrics,
    );
    let record = StructuralAdaptationIteration {
        schema_version: STRUCTURAL_ADAPTATION_ITERATION_SCHEMA_VERSION,
        iteration_index: index,
        previous_iteration_digest: input.previous.map(|p| p.canonical_digest()).transpose()?,
        source_solver_artifact_digest: input.source_artifact.canonical_digest,
        target_solver_artifact_digest: input.target_artifact.canonical_digest,
        adaptation_lineage_digest: input
            .lineage
            .canonical_digest()
            .map_err(|e| StructuralAdaptationIterationError::InvalidLineage(e.to_string()))?,
        transfer_map_digest: input
            .transfer_map
            .canonical_digest()
            .map_err(|e| StructuralAdaptationIterationError::InvalidLineage(e.to_string()))?,
        source_estimator: input.source_estimator.clone(),
        estimator: input.estimator.clone(),
        transfer_errors: input.transfer_errors.to_vec(),
        source_solver_result: input.source_solver_result.clone(),
        solver_result: input.solver_result.clone(),
        source_target_quantity: input.source_target_quantity.clone(),
        target_quantity: input.target_quantity.clone(),
        constraint_lineage: constraints,
        accuracy_plan: plan.clone(),
        accuracy_evaluation: evaluation,
        decision,
    };
    record.validate()?;
    record.validate_mesh_evidence(
        input.source_artifact,
        input.target_artifact,
        input.transfer_map,
        input.lineage,
    )?;
    if let Some(previous) = input.previous {
        record.validate_previous(previous)?;
    }
    Ok(record)
}

impl StructuralAdaptationIteration {
    pub fn validate(&self) -> Result<(), StructuralAdaptationIterationError> {
        use StructuralAdaptationIterationError as Error;
        if self.schema_version != STRUCTURAL_ADAPTATION_ITERATION_SCHEMA_VERSION
            || [
                self.source_solver_artifact_digest,
                self.target_solver_artifact_digest,
                self.adaptation_lineage_digest,
                self.transfer_map_digest,
            ]
            .contains(&StableDigest::ZERO)
            || self.source_solver_artifact_digest == self.target_solver_artifact_digest
            || self.source_estimator.solver_artifact_digest != self.source_solver_artifact_digest
            || self.estimator.solver_artifact_digest != self.target_solver_artifact_digest
            || self.previous_iteration_digest.is_some() != (self.iteration_index > 0)
            || self.previous_iteration_digest == Some(StableDigest::ZERO)
        {
            return Err(Error::InvalidIterationChain);
        }
        for estimate in [&self.source_estimator, &self.estimator] {
            estimate.validate().map_err(|_| Error::InvalidEstimator)?;
        }
        validate_transfer_errors(
            &self.transfer_errors,
            self.source_solver_artifact_digest,
            self.target_solver_artifact_digest,
        )?;
        for solver in [&self.source_solver_result, &self.solver_result] {
            validate_solver_result(solver)?;
        }
        for quantity in [&self.source_target_quantity, &self.target_quantity] {
            validate_target_quantity(quantity)?;
        }
        if self.source_target_quantity.quantity_id != self.target_quantity.quantity_id
            || self.source_target_quantity.quantity != self.target_quantity.quantity
        {
            return Err(Error::InvalidTargetQuantity);
        }
        self.constraint_lineage.validate()?;
        let metrics = decision::changes(
            self.source_estimator.total_error,
            self.estimator.total_error,
            self.source_target_quantity.value,
            self.target_quantity.value,
        )?;
        // Validate retained raw metrics before they can become accuracy evidence.
        if self.decision.target_absolute_change != metrics.1
            || self.decision.target_relative_change != metrics.2
        {
            return Err(Error::InvalidDecision);
        }
        let expected = accuracy::evaluate(accuracy::AccuracyInput {
            plan: &self.accuracy_plan,
            source_estimator: &self.source_estimator,
            estimator: &self.estimator,
            transfer_errors: &self.transfer_errors,
            solver_result: &self.solver_result,
            target_quantity: &self.target_quantity,
            absolute_change: metrics.1,
            relative_change: metrics.2,
        })?;
        if self.accuracy_evaluation != expected
            || self.decision
                != decision::decide(
                    &self.accuracy_plan,
                    &expected,
                    self.iteration_index,
                    self.source_solver_result.converged && self.solver_result.converged,
                    metrics,
                )
        {
            return Err(Error::InvalidDecision);
        }
        Ok(())
    }

    pub fn validate_against(
        &self,
        model: &runmat_analysis_core::AnalysisModel,
        source: &SolverMeshArtifact,
        target: &SolverMeshArtifact,
        transfer: &SolverMeshTransferMap,
        lineage: &SolverMeshAdaptationLineage,
    ) -> Result<(), StructuralAdaptationIterationError> {
        use StructuralAdaptationIterationError as Error;
        self.validate()?;
        self.validate_mesh_evidence(source, target, transfer, lineage)?;
        if self.constraint_lineage
            != StructuralAdaptationConstraintLineage::derive(model, source, target)?
        {
            return Err(Error::InvalidConstraints(
                "retained plans differ from canonical recompilation".into(),
            ));
        }
        Ok(())
    }

    fn validate_mesh_evidence(
        &self,
        source: &SolverMeshArtifact,
        target: &SolverMeshArtifact,
        transfer: &SolverMeshTransferMap,
        lineage: &SolverMeshAdaptationLineage,
    ) -> Result<(), StructuralAdaptationIterationError> {
        use StructuralAdaptationIterationError as Error;
        lineage
            .validate_against(source, target, transfer)
            .map_err(|e| Error::InvalidLineage(e.to_string()))?;
        validate_estimator_against(&self.source_estimator, source)?;
        validate_estimator_against(&self.estimator, target)?;
        validate_transfer_errors_against(&self.transfer_errors, source, target)?;
        if self.source_solver_artifact_digest != source.canonical_digest
            || self.target_solver_artifact_digest != target.canonical_digest
            || self.adaptation_lineage_digest
                != lineage
                    .canonical_digest()
                    .map_err(|e| Error::InvalidLineage(e.to_string()))?
            || self.transfer_map_digest
                != transfer
                    .canonical_digest()
                    .map_err(|e| Error::InvalidLineage(e.to_string()))?
        {
            return Err(Error::InvalidIterationChain);
        }
        Ok(())
    }

    pub fn validate_previous(
        &self,
        previous: &Self,
    ) -> Result<(), StructuralAdaptationIterationError> {
        previous.validate()?;
        self.validate()?;
        if previous.iteration_index.checked_add(1) != Some(self.iteration_index)
            || self.previous_iteration_digest != Some(previous.canonical_digest()?)
            || previous.target_solver_artifact_digest != self.source_solver_artifact_digest
            || previous.estimator != self.source_estimator
            || previous.solver_result != self.source_solver_result
            || previous.target_quantity != self.source_target_quantity
            || previous.accuracy_plan != self.accuracy_plan
            || previous.constraint_lineage.target_plan_digest
                != self.constraint_lineage.source_plan_digest
            || previous.decision.status != StructuralAdaptationDecisionStatus::Continue
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
fn codec_error(error: CanonicalCodecError) -> StructuralAdaptationIterationError {
    StructuralAdaptationIterationError::Codec(error.to_string())
}
#[cfg(test)]
pub(super) mod tests;
