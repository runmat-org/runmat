//! Original static result provenance, not certification reconstructed from a hash.
use super::{StructuralAdaptationSolverResult, StructuralTargetQuantity};
use crate::{
    contracts::FeaRunResult,
    execution::{SequenceIdentity, SolverResultIdentity, StepIdentity},
    results::observation::{
        resolve_structural_observation_plan, ObservationLimits, ObservationResult,
    },
};
use runmat_analysis_core::{
    AnalysisModel, ComplexRepresentation, ObservationId, ObservationValueShape,
};
use runmat_canonical_codec::{encode_contract_with_control, profile_serialized, CanonicalLimits};
use runmat_execution::Digest;
use runmat_meshing_core::{SolverMeshArtifact, StableDigest};
use serde::{Deserialize, Serialize};

mod iteration;
mod observation;
pub use iteration::{
    build_structural_adaptation_iteration_from_solved_results, StructuralAdaptationSolvedIteration,
    StructuralAdaptationSolvedIterationInput,
};

#[derive(Clone, Copy, Debug)]
pub struct StructuralAdaptationSourceLimits {
    pub canonical: CanonicalLimits,
    /// Additional canonical encoding and retained-field-copy budget, excluding owned inputs
    /// and constraint/observation-plan construction workspaces.
    pub maximum_materialization_bytes: u64,
    pub observations: ObservationLimits,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuralAdaptationSolvedSource {
    pub schema_version: u16,
    pub solver_artifact_digest: StableDigest,
    pub original_content_digest: Digest,
    pub solver: StructuralAdaptationSolverResult,
    pub target_quantity: StructuralTargetQuantity,
    pub observation: ObservationResult,
}

#[derive(Clone, Copy)]
pub struct StructuralAdaptationSolvedSourceInput<'a> {
    pub model: &'a AnalysisModel,
    pub mesh: &'a SolverMeshArtifact,
    pub run: &'a FeaRunResult,
    pub observation_id: &'a ObservationId,
    pub limits: StructuralAdaptationSourceLimits,
}

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum StructuralAdaptationSourceError {
    #[error("original structural adaptation source is invalid: {0}")]
    Invalid(String),
    #[error("original structural adaptation source exceeds materialization budget")]
    ResourceLimit,
    #[error("original structural adaptation source operation was cancelled")]
    Cancelled,
}
use StructuralAdaptationSourceError as Error;
fn invalid(error: impl std::fmt::Display) -> Error {
    if crate::is_fea_cancelled() {
        Error::Cancelled
    } else {
        Error::Invalid(error.to_string())
    }
}

impl StructuralAdaptationSolvedSource {
    pub fn derive(input: StructuralAdaptationSolvedSourceInput<'_>) -> Result<Self, Error> {
        if crate::is_fea_cancelled() {
            return Err(Error::Cancelled);
        }
        let StructuralAdaptationSolvedSourceInput {
            model,
            mesh,
            run,
            observation_id,
            limits,
        } = input;
        // Admit canonical encoding and retained-field copies before deriving constraints or
        // observation plans. Their own construction workspaces remain separate admission.
        let content = (model, mesh.canonical_digest, run);
        let profile = profile_serialized(&content, limits.canonical).map_err(invalid)?;
        let field_copy = profile_serialized(&run.fields, limits.canonical).map_err(invalid)?;
        let preliminary = profile
            .materialization_resident_bytes_estimate
            .checked_add(field_copy.materialization_resident_bytes_estimate)
            .ok_or(Error::ResourceLimit)?;
        if preliminary > limits.maximum_materialization_bytes {
            return Err(Error::ResourceLimit);
        }
        if crate::is_fea_cancelled() {
            return Err(Error::Cancelled);
        }
        // Solver identity indexes borrowed metadata in owned vectors. That work follows
        // admission of the original model/result and retained-field-copy profile.
        let numerical = (
            model.solver_identity_contract(),
            mesh.canonical_digest,
            &run.identity,
            &run.structural_static_convergence,
            &run.structural_static_balance,
            &run.structural_constraint_result,
            &run.structural_connector_result,
            &run.spatial_projection_evidence,
            &run.observations,
            &run.fields,
        );
        let numerical_profile =
            profile_serialized(&numerical, limits.canonical).map_err(invalid)?;
        let required = profile
            .materialization_resident_bytes_estimate
            .max(numerical_profile.materialization_resident_bytes_estimate)
            .checked_add(field_copy.materialization_resident_bytes_estimate)
            .ok_or(Error::ResourceLimit)?;
        if required > limits.maximum_materialization_bytes {
            return Err(Error::ResourceLimit);
        }
        if crate::is_fea_cancelled() {
            return Err(Error::Cancelled);
        }
        mesh.validate().map_err(invalid)?;
        run.validate_solver_formulation().map_err(invalid)?;
        let convergence = run
            .structural_static_convergence
            .as_ref()
            .ok_or_else(|| invalid("original static convergence is missing"))?;
        let normalized_residual = convergence
            .relative_residual
            .ok_or_else(|| invalid("original solved equation has no relative residual"))?;
        let constraints =
            crate::constraints::CanonicalStructuralConstraintPlan::compile(model, mesh)
                .map_err(invalid)?;
        run.structural_constraint_result
            .as_ref()
            .ok_or_else(|| invalid("missing executed constraints"))?
            .validate_against_model(&constraints, model, Some(mesh))
            .map_err(invalid)?;
        let step = model
            .sequence
            .steps
            .first()
            .ok_or_else(|| invalid("missing static step"))?;
        if model.sequence.steps.len() != 1 {
            return Err(invalid("source requires the original single static step"));
        }
        let definitions = crate::post::fields::structural_static_observable_field_definitions()
            .map_err(invalid)?
            .into_iter()
            .filter(|d| run.field(d.field_id()).is_some_and(|f| !f.is_empty()))
            .collect::<Vec<_>>();
        let plan = resolve_structural_observation_plan(
            step,
            model,
            Some(mesh),
            &constraints,
            &definitions,
            limits.observations,
        )
        .map_err(invalid)?;
        let observation = run
            .observations
            .iter()
            .find(|o| &o.observation_id == observation_id)
            .ok_or_else(|| invalid("the original solved target observation is missing"))?;
        if run
            .observations
            .iter()
            .filter(|o| &o.observation_id == observation_id)
            .count()
            != 1
        {
            return Err(invalid("duplicate original target observations"));
        }
        observation.validate().map_err(invalid)?;
        let resolved = plan
            .observations
            .iter()
            .find(|o| &o.spec.observation_id == observation_id)
            .ok_or_else(|| invalid("target observation is not authored"))?;
        if observation.plan_identity != plan.identity
            || observation.topology_digest != plan.topology_digest
            || observation.quantity != resolved.spec.quantity
            || observation.value_shape != ObservationValueShape::Scalar
            || observation.sample != crate::results::field::RecoveredFieldSample::Final
            || observation.complex_representation != ComplexRepresentation::Real
        {
            return Err(invalid(
                "original target requires the exact authored final real scalar observation",
            ));
        }
        let values = observation
            .field
            .as_host_f64()
            .ok_or_else(|| invalid("target observation must retain original host values"))?;
        if values.len() != 1 {
            return Err(invalid("target observation must reduce to one scalar"));
        }
        observation::verify(input, &plan, observation)?;
        let bytes = encode_contract_with_control(
            b"runmat-fea-adaptation-solved-source/v1\0",
            "original-static-result",
            &content,
            limits.canonical,
            &crate::progress::cancellation_token(),
        )
        .map_err(|e| {
            if crate::is_fea_cancelled() {
                Error::Cancelled
            } else {
                invalid(e)
            }
        })?;
        let original_content_digest = Digest::sha256(bytes);
        let sequence = SequenceIdentity::from_solver_contract(&model.sequence).map_err(invalid)?;
        let step = StepIdentity::from_solver_contract(sequence, step).map_err(invalid)?;
        // Logical numerical identity excludes backend/attempt diagnostics; full provenance
        // above still binds every original returned byte to this particular source.
        let numerical_bytes = encode_contract_with_control(
            b"runmat-fea-adaptation-solved-source/v1\0",
            "numerical-static-result",
            &numerical,
            limits.canonical,
            &crate::progress::cancellation_token(),
        )
        .map_err(|e| {
            if crate::is_fea_cancelled() {
                Error::Cancelled
            } else {
                invalid(e)
            }
        })?;
        let result_identity =
            SolverResultIdentity::from_step_result(step, Digest::sha256(numerical_bytes))
                .map_err(invalid)?;
        Ok(Self {
            schema_version: 1,
            solver_artifact_digest: mesh.canonical_digest,
            original_content_digest,
            solver: StructuralAdaptationSolverResult {
                result_identity,
                converged: convergence.converged,
                iteration_count: u64::from(convergence.iterations),
                normalized_residual,
            },
            target_quantity: StructuralTargetQuantity {
                quantity_id: observation_id.0.clone(),
                quantity: observation.quantity,
                value: values[0],
            },
            observation: observation.clone(),
        })
    }
    /// Cold provenance revalidation requires the original model, mesh and retained result.
    /// Deserializing this summary alone neither verifies its source nor repeats the solve.
    pub fn validate_against(
        &self,
        input: StructuralAdaptationSolvedSourceInput<'_>,
    ) -> Result<(), Error> {
        if *self != Self::derive(input)? {
            return Err(invalid(
                "retained source differs from original physical result",
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
