use runmat_analysis_fea::{execution::StructuralStudyWorkloads, study::CompiledAnalysisStudy};
use runmat_execution::Digest;
use runmat_meshing_core::{MeshingEvidence, SolverMeshArtifact};

use super::StructuralStudyNumericalResult;

mod acoustic;
mod acoustic_response;
mod electrical;
mod fluid;
mod magnetic;
mod maxwell;
mod nonlinear;
pub use acoustic_response::{
    AcousticResponseStudyCompletion, AcousticResponseStudyObjectSource, AcousticResponseStudySource,
};
pub use electrical::{
    ElectricalStudyCompletion, ElectricalStudyObjectSource, ElectricalStudySource,
};
pub use fluid::{FluidStudyCompletion, FluidStudyObjectSource, FluidStudySource};
pub use magnetic::{MagneticStudyCompletion, MagneticStudyObjectSource, MagneticStudySource};
pub use maxwell::{MaxwellStudyCompletion, MaxwellStudyObjectSource, MaxwellStudySource};
pub use nonlinear::{
    NonlinearSolidStudyCompletion, NonlinearSolidStudyObjectSource, NonlinearSolidStudySource,
};
pub(super) mod thermal;
pub use acoustic::{
    AcousticStudyCompletion, AcousticStudyEntryCompletion, AcousticStudyObjectSource,
    AcousticStudySource,
};
mod thermal_completion;
pub(super) mod thermal_observation_admission;
mod thermal_observations;
mod thermal_source;
pub use thermal_completion::ThermalStudyCompletion;
pub use thermal_observations::{
    ThermalObservationHistoryRoot, ThermalObservationObjectRoot, ThermalObservationPublication,
    ThermalObservationSampleRoot, ThermalObservationSeriesRoot,
};
pub use thermal_source::{ThermalStudyObjectSource, ThermalStudySource};

/// Placement-neutral completed structural study.
///
/// The compiled analysis definition, canonical mesh, governed meshing evidence, and numerical
/// result travel together. Local execution and a verified remote import must construct this same
/// type; paths, worker identities, retry state, and transport authority do not belong here.
#[derive(Debug, Clone, serde::Serialize)]
pub struct StructuralStudyResult {
    compiled: CompiledAnalysisStudy,
    mesh: SolverMeshArtifact,
    meshing_evidence: MeshingEvidence,
    workloads: StructuralStudyWorkloads,
    numerical: StructuralStudyNumericalResult,
}

impl StructuralStudyResult {
    pub fn new(
        compiled: CompiledAnalysisStudy,
        mesh: SolverMeshArtifact,
        meshing_evidence: MeshingEvidence,
        workloads: StructuralStudyWorkloads,
        numerical: StructuralStudyNumericalResult,
    ) -> Result<Self, String> {
        compiled
            .validate_against(&mesh)
            .map_err(|error| error.to_string())?;
        meshing_evidence
            .validate(&mesh)
            .map_err(|error| error.to_string())?;
        workloads.validate().map_err(|error| error.to_string())?;
        if workloads
            .authored_model()
            .map_err(|error| error.to_string())?
            != &compiled.model
        {
            return Err("structural study workloads differ from the compiled model".into());
        }
        validate_numerical_kind_and_steps(&workloads, &numerical)?;
        if let StructuralStudyNumericalResult::ThermalPrestress {
            thermal,
            thermal_source,
            structural,
        } = &numerical
        {
            thermal_source.authenticate(thermal)?;
            self::thermal::validate(&compiled, &mesh, &workloads, thermal, structural)?;
        }
        validate_numerical_evidence(&numerical)?;
        Ok(Self {
            compiled,
            mesh,
            meshing_evidence,
            workloads,
            numerical,
        })
    }

    pub fn definition_digest(&self) -> Result<Digest, String> {
        self.compiled
            .definition_digest()
            .map_err(|error| error.to_string())
    }

    pub const fn compiled(&self) -> &CompiledAnalysisStudy {
        &self.compiled
    }

    pub const fn mesh(&self) -> &SolverMeshArtifact {
        &self.mesh
    }

    pub const fn meshing_evidence(&self) -> &MeshingEvidence {
        &self.meshing_evidence
    }

    pub const fn workloads(&self) -> &StructuralStudyWorkloads {
        &self.workloads
    }

    pub const fn numerical(&self) -> &StructuralStudyNumericalResult {
        &self.numerical
    }

    pub fn into_parts(
        self,
    ) -> (
        CompiledAnalysisStudy,
        SolverMeshArtifact,
        MeshingEvidence,
        StructuralStudyWorkloads,
        StructuralStudyNumericalResult,
    ) {
        (
            self.compiled,
            self.mesh,
            self.meshing_evidence,
            self.workloads,
            self.numerical,
        )
    }
}

fn validate_numerical_evidence(numerical: &StructuralStudyNumericalResult) -> Result<(), String> {
    match numerical {
        StructuralStudyNumericalResult::ThermalPrestress { structural, .. } => {
            validate_numerical_evidence(structural)?
        }
        StructuralStudyNumericalResult::LinearSequence(result) => {
            for step in &result.steps {
                step.numerical_receipt
                    .validate_against(step.solver_result_identity, &step.result)
                    .map_err(|error| error.to_string())?;
            }
            for state in &result.states {
                state.validate().map_err(|error| error.to_string())?;
            }
        }
        StructuralStudyNumericalResult::StructuralModal(result) => result
            .numerical_receipt
            .validate_against(result.numerical_receipt.source_result(), &result.result)
            .map_err(|error| error.to_string())?,
        StructuralStudyNumericalResult::StructuralModalTransient(result) => {
            result
                .modal_numerical_receipt
                .validate_against(
                    result.modal_numerical_receipt.source_result(),
                    &result.modal_result,
                )
                .map_err(|error| error.to_string())?;
            result
                .transient_numerical_receipt
                .validate_for_result(&result.transient_result)
                .map_err(|error| error.to_string())?;
            result
                .basis_state
                .validate()
                .map_err(|error| error.to_string())?;
        }
        StructuralStudyNumericalResult::StructuralModalResponseSpectrum(result) => {
            result
                .modal_numerical_receipt
                .validate_against(
                    result.modal_numerical_receipt.source_result(),
                    &result.modal_result,
                )
                .map_err(|error| error.to_string())?;
            result
                .response_numerical_receipt
                .validate_against(
                    result.response_numerical_receipt.source_result(),
                    &result.response_result,
                )
                .map_err(|error| error.to_string())?;
            result
                .basis_state
                .validate()
                .map_err(|error| error.to_string())?;
            result
                .envelope_state
                .validate()
                .map_err(|error| error.to_string())?;
        }
        StructuralStudyNumericalResult::StructuralRandomResponse(result) => {
            result
                .modal_numerical_receipt
                .validate_against(
                    result.modal_numerical_receipt.source_result(),
                    &result.modal_result,
                )
                .map_err(|error| error.to_string())?;
            result
                .response_numerical_receipt
                .validate_against(
                    result.response_numerical_receipt.source_result(),
                    &result.response_result,
                )
                .map_err(|error| error.to_string())?;
            result
                .basis_state
                .validate()
                .map_err(|error| error.to_string())?;
            result
                .statistics_state
                .validate()
                .map_err(|error| error.to_string())?;
        }
        StructuralStudyNumericalResult::StructuralBuckling(result) => {
            validate_linear_producer(
                &result.preload_numerical_receipt,
                result.preload_solver_result_identity,
                &result.preload_result,
            )?;
            result
                .buckling_numerical_receipt
                .validate_against(
                    result.buckling_numerical_receipt.source_result(),
                    &result.buckling_result,
                )
                .map_err(|error| error.to_string())?;
            result
                .prestress_state
                .validate()
                .map_err(|error| error.to_string())?;
        }
        StructuralStudyNumericalResult::StructuralPrestressedModal(result) => {
            validate_linear_producer(
                &result.preload_numerical_receipt,
                result.preload_solver_result_identity,
                &result.preload_result,
            )?;
            result
                .modal_numerical_receipt
                .validate_against(
                    result.modal_numerical_receipt.source_result(),
                    &result.modal_result,
                )
                .map_err(|error| error.to_string())?;
            result
                .prestress_state
                .validate()
                .map_err(|error| error.to_string())?;
        }
        StructuralStudyNumericalResult::StructuralPrestressedFrequency(result) => {
            validate_linear_producer(
                &result.preload_numerical_receipt,
                result.preload_solver_result_identity,
                &result.preload_result,
            )?;
            result
                .prestress_state
                .validate()
                .map_err(|error| error.to_string())?;
            if let super::StructuralStudyPrestressedFrequencyReductionResult::ModalReduced(
                reduction,
            ) = &result.reduction
            {
                reduction
                    .modal_numerical_receipt
                    .validate_against(
                        reduction.modal_numerical_receipt.source_result(),
                        &reduction.modal_result,
                    )
                    .map_err(|error| error.to_string())?;
                reduction
                    .basis_state
                    .validate()
                    .map_err(|error| error.to_string())?;
            }
        }
        StructuralStudyNumericalResult::StructuralModalFrequency(result) => {
            result
                .modal_numerical_receipt
                .validate_against(
                    result.modal_numerical_receipt.source_result(),
                    &result.modal_result,
                )
                .map_err(|error| error.to_string())?;
            result
                .basis_state
                .validate()
                .map_err(|error| error.to_string())?;
        }
        StructuralStudyNumericalResult::StructuralTransient(result) => result
            .numerical_receipt
            .validate_for_result(&result.result)
            .map_err(|error| error.to_string())?,
        StructuralStudyNumericalResult::StructuralFrequency(_) => {}
    }
    Ok(())
}

fn validate_linear_producer(
    receipt: &runmat_analysis_fea::execution::StructuralLinearNumericalReceiptArtifact,
    identity: runmat_analysis_fea::execution::SolverResultIdentity,
    result: &runmat_analysis_fea::execution::StructuralLinearWorkloadResult,
) -> Result<(), String> {
    receipt
        .validate_against(identity, result)
        .map_err(|error| error.to_string())
}

fn validate_numerical_kind_and_steps(
    workloads: &StructuralStudyWorkloads,
    numerical: &StructuralStudyNumericalResult,
) -> Result<(), String> {
    let expected = workloads.step_ids().map_err(|error| error.to_string())?;
    let actual = match numerical {
        StructuralStudyNumericalResult::ThermalPrestress {
            thermal,
            structural,
            ..
        } => {
            let prefix = workloads
                .prestress_prefix()
                .map_err(|error| error.to_string())?;
            let mut steps = vec![thermal.workload.step_id.clone()];
            let (preload, consumers) = thermal::structural_steps(structural)?;
            if preload != prefix.preload.step_id || consumers != prefix.consumer_step_ids {
                return Err(
                    "thermal prestress structural child differs from authored suffix".into(),
                );
            }
            steps.push(preload);
            steps.extend(consumers);
            steps
        }
        StructuralStudyNumericalResult::LinearSequence(result) => result
            .steps
            .iter()
            .map(|step| step.step_id.clone())
            .collect(),
        StructuralStudyNumericalResult::StructuralModal(result) => vec![result.step_id.clone()],
        StructuralStudyNumericalResult::StructuralModalTransient(result) => {
            vec![
                result.modal_step_id.clone(),
                result.transient_step_id.clone(),
            ]
        }
        StructuralStudyNumericalResult::StructuralModalFrequency(result) => {
            vec![
                result.modal_step_id.clone(),
                result.frequency_step_id.clone(),
            ]
        }
        StructuralStudyNumericalResult::StructuralModalResponseSpectrum(result) => {
            vec![
                result.modal_step_id.clone(),
                result.response_step_id.clone(),
            ]
        }
        StructuralStudyNumericalResult::StructuralRandomResponse(result) => {
            vec![
                result.modal_step_id.clone(),
                result.response_step_id.clone(),
            ]
        }
        StructuralStudyNumericalResult::StructuralBuckling(result) => {
            vec![
                result.preload_step_id.clone(),
                result.buckling_step_id.clone(),
            ]
        }
        StructuralStudyNumericalResult::StructuralPrestressedModal(result) => {
            vec![result.preload_step_id.clone(), result.modal_step_id.clone()]
        }
        StructuralStudyNumericalResult::StructuralPrestressedFrequency(result) => {
            let mut steps = vec![result.preload_step_id.clone()];
            if let super::StructuralStudyPrestressedFrequencyReductionResult::ModalReduced(
                reduction,
            ) = &result.reduction
            {
                steps.push(reduction.modal_step_id.clone());
            }
            steps.push(result.frequency_step_id.clone());
            steps
        }
        StructuralStudyNumericalResult::StructuralTransient(result) => vec![result.step_id.clone()],
        StructuralStudyNumericalResult::StructuralFrequency(result) => {
            vec![result.step_id().clone()]
        }
    };
    if actual != expected {
        return Err("structural numerical result differs from its exact workload steps".into());
    }
    let compatible = matches!(
        (workloads, numerical),
        (
            StructuralStudyWorkloads::LinearSequence { .. },
            StructuralStudyNumericalResult::LinearSequence(_)
        ) | (
            StructuralStudyWorkloads::StructuralTransient { .. },
            StructuralStudyNumericalResult::StructuralTransient(_)
        ) | (
            StructuralStudyWorkloads::StructuralFrequency { .. },
            StructuralStudyNumericalResult::StructuralFrequency(_)
        ) | (
            StructuralStudyWorkloads::StructuralModal { .. },
            StructuralStudyNumericalResult::StructuralModal(_)
        ) | (
            StructuralStudyWorkloads::HeterogeneousSequence { .. },
            StructuralStudyNumericalResult::StructuralModalTransient(_)
                | StructuralStudyNumericalResult::StructuralModalFrequency(_)
                | StructuralStudyNumericalResult::StructuralModalResponseSpectrum(_)
                | StructuralStudyNumericalResult::StructuralRandomResponse(_)
                | StructuralStudyNumericalResult::StructuralBuckling(_)
                | StructuralStudyNumericalResult::StructuralPrestressedModal(_)
                | StructuralStudyNumericalResult::StructuralPrestressedFrequency(_)
                | StructuralStudyNumericalResult::ThermalPrestress { .. }
        )
    );
    if !compatible {
        return Err("structural numerical result kind differs from its exact workloads".into());
    }
    Ok(())
}
