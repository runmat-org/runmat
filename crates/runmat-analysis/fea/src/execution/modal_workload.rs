use runmat_analysis_core::{
    validate_model_sequence, AnalysisModel, AnalysisProcedure, FormulationRequirement,
    IndependentDomain, PhysicsFamily, ProcedureStep, ProcedureStepId, RestartPolicy, SolutionSpace,
    StateCompleteness, StateKind, StateOutput, StatePersistence, StateSource, StateUse,
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
    CanonicalLimits::new(64 * 1024 * 1024, 1_000_000, 65_536, 64);

/// Authored prestressed-modal intent before its exact preload state has been published.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuralPrestressedModalTemplate {
    pub schema_version: u16,
    pub model: AnalysisModel,
    pub step_id: ProcedureStepId,
    pub backend: ComputeBackend,
    pub formulation_requirement: FormulationRequirement,
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
            formulation_requirement: self.formulation_requirement,
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
    formulation_requirement: FormulationRequirement,
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
    formulation_requirement: FormulationRequirement,
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
            formulation_requirement: wire.formulation_requirement,
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
    pub formulation_requirement: FormulationRequirement,
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
            formulation_requirement: input.formulation_requirement,
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

    pub const fn formulation_requirement(&self) -> FormulationRequirement {
        self.formulation_requirement
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
                self.formulation_requirement,
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
mod tests {
    use super::*;

    fn workload() -> StructuralModalWorkload {
        let mut model =
            crate::fixtures::fixture_model(crate::fixtures::FixtureId::CantileverLinearStatic);
        let truncation = ModalTruncationPolicy::requested(3);
        let mut step = ProcedureStep::structural_modal("modes");
        step.independent_domain = IndependentDomain::Eigenvalue {
            requested_count: truncation.requested_mode_count(),
        };
        model.sequence.steps = vec![step];
        StructuralModalWorkload::new(StructuralModalWorkloadInput {
            model,
            step_id: ProcedureStepId("modes".into()),
            backend: ComputeBackend::Cpu,
            formulation_requirement: FormulationRequirement::Experimental,
            observation_execution: ObservationExecutionPolicy::scalable(),
            truncation,
            convergence: ModalConvergencePolicy::exploratory(),
            residual_vector_directions: Vec::new(),
            prestress_transition: None,
        })
        .unwrap()
    }

    #[test]
    fn modal_workload_round_trips_with_convergent_identity() {
        let workload = workload();
        let restored =
            StructuralModalWorkload::from_canonical_bytes(&workload.canonical_bytes().unwrap())
                .unwrap();
        assert_eq!(restored, workload);
        assert_eq!(
            workload.modal_basis_output().unwrap(),
            &workload.procedure_step().unwrap().state_outputs[0]
        );
    }

    #[test]
    fn residual_vector_directions_are_canonical_workload_inputs() {
        let base = workload();
        let direction = runmat_analysis_core::SpectrumDirection {
            frame: runmat_analysis_core::CoordinateFrame::Global,
            vector: [1.0, 0.5, 0.25],
        };
        let enriched = StructuralModalWorkload::new(StructuralModalWorkloadInput {
            model: base.model().clone(),
            step_id: base.step_id().clone(),
            backend: base.backend(),
            formulation_requirement: base.formulation_requirement(),
            observation_execution: base.observation_execution(),
            truncation: base.truncation(),
            convergence: base.convergence(),
            residual_vector_directions: vec![direction.clone()],
            prestress_transition: base.prestress_transition().cloned(),
        })
        .unwrap();
        assert_ne!(enriched.identity(), base.identity());
        assert_eq!(
            enriched.residual_vector_directions(),
            std::slice::from_ref(&direction)
        );
        assert_eq!(
            StructuralModalWorkload::from_canonical_bytes(&enriched.canonical_bytes().unwrap())
                .unwrap(),
            enriched
        );

        assert!(StructuralModalWorkload::new(StructuralModalWorkloadInput {
            model: base.model().clone(),
            step_id: base.step_id().clone(),
            backend: base.backend(),
            formulation_requirement: base.formulation_requirement(),
            observation_execution: base.observation_execution(),
            truncation: base.truncation(),
            convergence: base.convergence(),
            residual_vector_directions: vec![
                direction.clone(),
                runmat_analysis_core::SpectrumDirection {
                    frame: runmat_analysis_core::CoordinateFrame::Global,
                    vector: [2.0, 1.0, 0.5],
                },
            ],
            prestress_transition: base.prestress_transition().cloned(),
        })
        .is_err());
    }

    #[test]
    fn modal_workload_rejects_an_unpublishable_basis_declaration() {
        let valid = workload();
        let invalid_models = [
            {
                let mut model = valid.model().clone();
                model.sequence.steps[0].state_outputs.clear();
                model
            },
            {
                let mut model = valid.model().clone();
                let duplicate = model.sequence.steps[0].state_outputs[0].clone();
                model.sequence.steps[0].state_outputs.push(duplicate);
                model
            },
            {
                let mut model = valid.model().clone();
                model.sequence.steps[0].state_outputs[0].kind = StateKind::StructuralPrimary;
                model
            },
            {
                let mut model = valid.model().clone();
                model.sequence.steps[0].state_outputs[0].completeness = StateCompleteness::Partial;
                model
            },
            {
                let mut model = valid.model().clone();
                model.sequence.steps[0].state_outputs[0].persistence =
                    StatePersistence::CheckpointOnly;
                model
            },
        ];

        for model in invalid_models {
            assert!(StructuralModalWorkload::new(StructuralModalWorkloadInput {
                model,
                step_id: valid.step_id().clone(),
                backend: valid.backend(),
                formulation_requirement: valid.formulation_requirement(),
                observation_execution: valid.observation_execution(),
                truncation: valid.truncation(),
                convergence: valid.convergence(),
                residual_vector_directions: valid.residual_vector_directions().to_vec(),
                prestress_transition: valid.prestress_transition().cloned(),
            })
            .is_err());
        }
    }

    #[test]
    fn modal_workload_accepts_a_basis_retained_until_dependents_complete() {
        let valid = workload();
        let mut model = valid.model().clone();
        model.sequence.steps[0].state_outputs[0].persistence =
            StatePersistence::UntilDependentsComplete;

        let workload = StructuralModalWorkload::new(StructuralModalWorkloadInput {
            model,
            step_id: valid.step_id().clone(),
            backend: valid.backend(),
            formulation_requirement: valid.formulation_requirement(),
            observation_execution: valid.observation_execution(),
            truncation: valid.truncation(),
            convergence: valid.convergence(),
            residual_vector_directions: valid.residual_vector_directions().to_vec(),
            prestress_transition: valid.prestress_transition().cloned(),
        })
        .unwrap();
        assert_eq!(
            workload.modal_basis_output().unwrap().persistence,
            StatePersistence::UntilDependentsComplete
        );
    }

    #[test]
    fn modal_workload_rejects_a_mismatched_requested_eigenspace() {
        let workload = workload();
        let mut model = workload.model().clone();
        model.sequence.steps[0].independent_domain =
            IndependentDomain::Eigenvalue { requested_count: 2 };
        assert!(StructuralModalWorkload::new(StructuralModalWorkloadInput {
            model,
            step_id: workload.step_id().clone(),
            backend: workload.backend(),
            formulation_requirement: workload.formulation_requirement(),
            observation_execution: workload.observation_execution(),
            truncation: workload.truncation(),
            convergence: workload.convergence(),
            residual_vector_directions: workload.residual_vector_directions().to_vec(),
            prestress_transition: workload.prestress_transition().cloned(),
        })
        .is_err());
    }

    #[test]
    fn model_rebinding_regenerates_the_workload_identity() {
        let mut workload = workload();
        let identity = workload.identity();
        let mut model = workload.model().clone();
        model.geometry_revision += 1;
        workload.replace_model(model).unwrap();
        assert_ne!(workload.identity(), identity);
    }
}
