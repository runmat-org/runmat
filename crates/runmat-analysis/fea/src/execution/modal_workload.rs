use runmat_analysis_core::{
    validate_model_sequence, AnalysisModel, AnalysisProcedure, FormulationRequirement,
    IndependentDomain, PhysicsFamily, ProcedureStep, ProcedureStepId, RestartPolicy, SolutionSpace,
    StateCompleteness, StateKind, StateOutput, StatePersistence,
};
use runmat_canonical_codec::{decode_contract, encode_contract, CanonicalLimits};
use serde::{Deserialize, Serialize};

use crate::{
    contracts::ComputeBackend,
    pipeline::modal_governance::{ModalConvergencePolicy, ModalTruncationPolicy},
    results::observation::ObservationExecutionPolicy,
};

use super::{AnalysisExecutionError, AnalysisExecutionResult, StructuralModalWorkloadIdentity};

const MODAL_WORKLOAD_PREFIX: &[u8] = b"runmat-structural-modal-workload/v1\0";
const MODAL_WORKLOAD_LIMITS: CanonicalLimits =
    CanonicalLimits::new(64 * 1024 * 1024, 1_000_000, 65_536, 64);

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
        if self.schema_version != 1 {
            return Err(AnalysisExecutionError::Invalid(
                "unsupported structural modal workload schema".into(),
            ));
        }
        validate_model_sequence(&self.model)
            .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?;
        self.truncation
            .validate()
            .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?;
        self.convergence
            .validate()
            .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?;
        let step = self.procedure_step()?;
        self.modal_basis_output()?;
        let IndependentDomain::Eigenvalue { requested_count } = &step.independent_domain else {
            return Err(AnalysisExecutionError::Invalid(
                "structural modal extraction requires an eigenvalue domain".into(),
            ));
        };
        if step.family != PhysicsFamily::Structural
            || step.procedure != AnalysisProcedure::ModalExtraction
            || step.solution_space != SolutionSpace::Direct
            || !step.state_inputs.is_empty()
            || !step.activations.is_empty()
            || step.restart != RestartPolicy::Fresh
            || *requested_count != self.truncation.requested_mode_count()
        {
            return Err(AnalysisExecutionError::Invalid(
                "structural modal workload must exactly describe one fresh direct modal-extraction step and its requested eigenspace".into(),
            ));
        }
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
            ),
        )
    }
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
