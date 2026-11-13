use runmat_analysis_core::{
    validate_model_sequence, AnalysisModel, AnalysisProcedure, IndependentDomain, PhysicsFamily,
    ProcedureStep, ProcedureStepId, RestartPolicy, SolutionSpace, StateCompleteness, StateKind,
    StateOutput, StatePersistence, StateSource, StateUse,
};
use runmat_canonical_codec::{decode_contract, encode_contract, CanonicalLimits};
use serde::{Deserialize, Serialize};

use crate::{
    contracts::ComputeBackend,
    pipeline::modal_governance::{ModalConvergencePolicy, ModalTruncationPolicy},
    results::observation::ObservationExecutionPolicy,
};

use super::{
    AnalysisExecutionError, AnalysisExecutionResult, CanonicalStateTransition, SequenceIdentity,
    StepIdentity, StructuralModalWorkloadIdentity,
};

const MODAL_WORKLOAD_PREFIX: &[u8] = b"runmat-structural-modal-workload/v1\0";
const MODAL_WORKLOAD_LIMITS: CanonicalLimits =
    crate::contracts::canonical_digest::wire_bounded(CanonicalLimits::new(0, 0, 65_536, 64));

/// Authored prestressed-modal intent before its exact preload state has been published.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuralPrestressedModalTemplate {
    pub schema_version: u16,
    pub model: AnalysisModel,
    pub step_id: ProcedureStepId,
    pub backend: ComputeBackend,
    pub observation_execution: ObservationExecutionPolicy,
    pub truncation: ModalTruncationPolicy,
    pub convergence: ModalConvergencePolicy,
    pub residual_vector_directions: Vec<runmat_analysis_core::SpectrumDirection>,
}

impl StructuralPrestressedModalTemplate {
    pub fn validate(&self) -> AnalysisExecutionResult<()> {
        let step = validate_modal_step(
            self.schema_version,
            &self.model,
            &self.step_id,
            self.truncation,
            self.convergence,
        )?;
        if step.procedure != AnalysisProcedure::PrestressedModalExtraction
            || !is_prestress_input(step)
        {
            return Err(AnalysisExecutionError::Invalid(
                "prestressed modal extraction requires one upstream geometric-stiffness input"
                    .into(),
            ));
        }
        validate_residual_vector_directions(&self.residual_vector_directions)?;
        Ok(())
    }

    pub fn bind(
        self,
        prestress_transition: CanonicalStateTransition,
    ) -> AnalysisExecutionResult<StructuralModalWorkload> {
        self.validate()?;
        StructuralModalWorkload::new(StructuralModalWorkloadInput {
            model: self.model,
            step_id: self.step_id,
            backend: self.backend,
            observation_execution: self.observation_execution,
            truncation: self.truncation,
            convergence: self.convergence,
            residual_vector_directions: self.residual_vector_directions,
            prestress_transition: Some(prestress_transition),
        })
    }
}

/// Path-free numerical authority for one structural modal-extraction step.
///
/// The canonical solver mesh remains a separate numerical object so it can be cached and moved
/// independently. Placement, retries, task resources, and artifact locations never enter this
/// contract or its identity.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, try_from = "StructuralModalWorkloadWire")]
pub struct StructuralModalWorkload {
    schema_version: u16,
    identity: StructuralModalWorkloadIdentity,
    model: AnalysisModel,
    step_id: ProcedureStepId,
    backend: ComputeBackend,
    observation_execution: ObservationExecutionPolicy,
    truncation: ModalTruncationPolicy,
    convergence: ModalConvergencePolicy,
    residual_vector_directions: Vec<runmat_analysis_core::SpectrumDirection>,
    prestress_transition: Option<CanonicalStateTransition>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct StructuralModalWorkloadWire {
    schema_version: u16,
    identity: StructuralModalWorkloadIdentity,
    model: AnalysisModel,
    step_id: ProcedureStepId,
    backend: ComputeBackend,
    observation_execution: ObservationExecutionPolicy,
    truncation: ModalTruncationPolicy,
    convergence: ModalConvergencePolicy,
    residual_vector_directions: Vec<runmat_analysis_core::SpectrumDirection>,
    prestress_transition: Option<CanonicalStateTransition>,
}

impl TryFrom<StructuralModalWorkloadWire> for StructuralModalWorkload {
    type Error = AnalysisExecutionError;

    fn try_from(wire: StructuralModalWorkloadWire) -> Result<Self, Self::Error> {
        let workload = Self {
            schema_version: wire.schema_version,
            identity: wire.identity,
            model: wire.model,
            step_id: wire.step_id,
            backend: wire.backend,
            observation_execution: wire.observation_execution,
            truncation: wire.truncation,
            convergence: wire.convergence,
            residual_vector_directions: wire.residual_vector_directions,
            prestress_transition: wire.prestress_transition,
        };
        workload.validate()?;
        Ok(workload)
    }
}

pub struct StructuralModalWorkloadInput {
    pub model: AnalysisModel,
    pub step_id: ProcedureStepId,
    pub backend: ComputeBackend,
    pub observation_execution: ObservationExecutionPolicy,
    pub truncation: ModalTruncationPolicy,
    pub convergence: ModalConvergencePolicy,
    pub residual_vector_directions: Vec<runmat_analysis_core::SpectrumDirection>,
    pub prestress_transition: Option<CanonicalStateTransition>,
}

impl StructuralModalWorkload {
    pub fn new(input: StructuralModalWorkloadInput) -> AnalysisExecutionResult<Self> {
        let mut workload = Self {
            schema_version: 1,
            identity: StructuralModalWorkloadIdentity::derive(
                "structural-modal-workload-placeholder",
                &1_u8,
            )?,
            model: input.model,
            step_id: input.step_id,
            backend: input.backend,
            observation_execution: input.observation_execution,
            truncation: input.truncation,
            convergence: input.convergence,
            residual_vector_directions: input.residual_vector_directions,
            prestress_transition: input.prestress_transition,
        };
        workload.identity = workload.derive_identity()?;
        workload.validate()?;
        Ok(workload)
    }

    pub const fn identity(&self) -> StructuralModalWorkloadIdentity {
        self.identity
    }

    pub fn model(&self) -> &AnalysisModel {
        &self.model
    }

    pub fn step_id(&self) -> &ProcedureStepId {
        &self.step_id
    }

    pub const fn backend(&self) -> ComputeBackend {
        self.backend
    }

    pub const fn observation_execution(&self) -> ObservationExecutionPolicy {
        self.observation_execution
    }

    pub const fn truncation(&self) -> ModalTruncationPolicy {
        self.truncation
    }

    pub const fn convergence(&self) -> ModalConvergencePolicy {
        self.convergence
    }

    pub fn residual_vector_directions(&self) -> &[runmat_analysis_core::SpectrumDirection] {
        &self.residual_vector_directions
    }

    pub const fn prestress_transition(&self) -> Option<&CanonicalStateTransition> {
        self.prestress_transition.as_ref()
    }

    pub fn replace_model(&mut self, model: AnalysisModel) -> AnalysisExecutionResult<()> {
        self.model = model;
        self.identity = self.derive_identity()?;
        self.validate()
    }

    pub fn procedure_step(&self) -> AnalysisExecutionResult<&ProcedureStep> {
        let mut matches = self
            .model
            .sequence
            .steps
            .iter()
            .filter(|step| step.step_id == self.step_id);
        let step = matches.next().ok_or_else(|| {
            AnalysisExecutionError::Invalid(format!(
                "structural modal workload references absent step `{}`",
                self.step_id.as_str()
            ))
        })?;
        if matches.next().is_some() {
            return Err(AnalysisExecutionError::Invalid(format!(
                "structural modal workload references duplicate step `{}`",
                self.step_id.as_str()
            )));
        }
        Ok(step)
    }

    /// Returns the one complete, publishable modal basis produced by this workload.
    ///
    /// Modal extraction is the numerical authority for a reusable basis. Keeping its output
    /// declaration invariant here ensures local execution, remote execution, and sequence
    /// coordination all admit the same semantic state contract.
    pub fn modal_basis_output(&self) -> AnalysisExecutionResult<&StateOutput> {
        let [output] = self.procedure_step()?.state_outputs.as_slice() else {
            return Err(AnalysisExecutionError::Invalid(
                "structural modal extraction must declare exactly one modal-basis output".into(),
            ));
        };
        if output.kind != StateKind::ModalBasis
            || output.completeness != StateCompleteness::Complete
            || output.persistence == StatePersistence::CheckpointOnly
        {
            return Err(AnalysisExecutionError::Invalid(
                "structural modal extraction requires one complete publishable modal-basis output"
                    .into(),
            ));
        }
        Ok(output)
    }

    pub fn validate(&self) -> AnalysisExecutionResult<()> {
        let step = validate_modal_step(
            self.schema_version,
            &self.model,
            &self.step_id,
            self.truncation,
            self.convergence,
        )?;
        let procedure_is_valid = match step.procedure {
            AnalysisProcedure::ModalExtraction => {
                step.state_inputs.is_empty() && self.prestress_transition.is_none()
            }
            AnalysisProcedure::PrestressedModalExtraction => {
                is_prestress_input(step)
                    && self
                        .prestress_transition
                        .as_ref()
                        .is_some_and(|transition| {
                            let sequence =
                                SequenceIdentity::from_solver_contract(&self.model.sequence);
                            let consumer = sequence.and_then(|sequence| {
                                StepIdentity::from_solver_contract(sequence, step)
                            });
                            transition.validate().is_ok()
                                && consumer.is_ok_and(|consumer| {
                                    transition.consumer == consumer
                                        && transition.consumer_step_id == self.step_id
                                })
                        })
            }
            _ => false,
        };
        if !procedure_is_valid {
            return Err(AnalysisExecutionError::Invalid(
                "structural modal workload must exactly describe one fresh direct modal-extraction step and its requested eigenspace".into(),
            ));
        }
        validate_residual_vector_directions(&self.residual_vector_directions)?;
        if self.identity != self.derive_identity()? {
            return Err(AnalysisExecutionError::Identity(
                "structural modal workload identity does not converge",
            ));
        }
        Ok(())
    }

    pub fn canonical_bytes(&self) -> AnalysisExecutionResult<Vec<u8>> {
        self.validate()?;
        Ok(encode_contract(
            MODAL_WORKLOAD_PREFIX,
            "structural-modal-workload",
            self,
            MODAL_WORKLOAD_LIMITS,
        )?)
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> AnalysisExecutionResult<Self> {
        Ok(decode_contract(
            MODAL_WORKLOAD_PREFIX,
            "structural-modal-workload",
            bytes,
            MODAL_WORKLOAD_LIMITS,
        )?)
    }

    fn derive_identity(&self) -> AnalysisExecutionResult<StructuralModalWorkloadIdentity> {
        StructuralModalWorkloadIdentity::derive(
            "structural-modal-workload",
            &(
                self.schema_version,
                &self.model,
                &self.step_id,
                self.backend,
                self.observation_execution,
                self.truncation,
                self.convergence,
                &self.residual_vector_directions,
                &self.prestress_transition,
            ),
        )
    }
}

fn validate_residual_vector_directions(
    directions: &[runmat_analysis_core::SpectrumDirection],
) -> AnalysisExecutionResult<()> {
    crate::pipeline::modal_residual::canonical_residual_vector_directions(directions)
        .map(|_| ())
        .map_err(AnalysisExecutionError::Invalid)
}

fn validate_modal_step<'a>(
    schema_version: u16,
    model: &'a AnalysisModel,
    step_id: &ProcedureStepId,
    truncation: ModalTruncationPolicy,
    convergence: ModalConvergencePolicy,
) -> AnalysisExecutionResult<&'a ProcedureStep> {
    if schema_version != 1 {
        return Err(AnalysisExecutionError::Invalid(
            "unsupported structural modal workload schema".into(),
        ));
    }
    validate_model_sequence(model)
        .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?;
    truncation
        .validate()
        .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?;
    convergence
        .validate()
        .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?;
    let mut matches = model
        .sequence
        .steps
        .iter()
        .filter(|step| step.step_id == *step_id);
    let step = matches.next().ok_or_else(|| {
        AnalysisExecutionError::Invalid(format!(
            "structural modal workload references absent step `{}`",
            step_id.as_str()
        ))
    })?;
    if matches.next().is_some() {
        return Err(AnalysisExecutionError::Invalid(format!(
            "structural modal workload references duplicate step `{}`",
            step_id.as_str()
        )));
    }
    let IndependentDomain::Eigenvalue { requested_count } = step.independent_domain else {
        return Err(AnalysisExecutionError::Invalid(
            "structural modal extraction requires an eigenvalue domain".into(),
        ));
    };
    let [output] = step.state_outputs.as_slice() else {
        return Err(AnalysisExecutionError::Invalid(
            "structural modal extraction must declare exactly one modal-basis output".into(),
        ));
    };
    if requested_count != truncation.requested_mode_count() {
        return Err(AnalysisExecutionError::Invalid(format!(
            "structural modal step requests {requested_count} modes but its solver policy requests {}",
            truncation.requested_mode_count()
        )));
    }
    if step.family != PhysicsFamily::Structural
        || step.solution_space != SolutionSpace::Direct
        || step.restart != RestartPolicy::Fresh
        || output.kind != StateKind::ModalBasis
        || output.completeness != StateCompleteness::Complete
        || output.persistence == StatePersistence::CheckpointOnly
    {
        return Err(AnalysisExecutionError::Invalid(
            "structural modal workload has an invalid direct eigenspace or output contract".into(),
        ));
    }
    Ok(step)
}

fn is_prestress_input(step: &ProcedureStep) -> bool {
    matches!(
        step.state_inputs.as_slice(),
        [input]
            if input.use_as == StateUse::Prestress
                && input.required_kind == StateKind::GeometricStiffness
                && matches!(input.source, StateSource::Upstream { .. })
    )
}

#[cfg(test)]
mod tests;
