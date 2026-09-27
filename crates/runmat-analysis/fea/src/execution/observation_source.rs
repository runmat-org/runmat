//! Borrowed producing workloads own structural observation policy and authored intent.
//!
//! Callers authenticate this authority by binding the workload to its original numerical
//! receipt and inputs. Selecting a variant or validating a workload alone is not receipt
//! attestation; this adapter only delegates to the existing workload owners.

use runmat_analysis_core::{AnalysisModel, FormulationRequirement, ProcedureStep};

use super::{
    AnalysisExecutionResult, StructuralFrequencyWorkload, StructuralLinearWorkload,
    StructuralModalResponseSpectrumWorkload, StructuralRandomResponseWorkload,
};
use crate::results::observation::ObservationExecutionPolicy;

#[derive(Clone, Copy, Debug)]
pub enum StructuralObservationSource<'a> {
    Linear(&'a StructuralLinearWorkload),
    Frequency(&'a StructuralFrequencyWorkload),
    ModalResponseSpectrum(&'a StructuralModalResponseSpectrumWorkload),
    RandomResponse(&'a StructuralRandomResponseWorkload),
}

impl<'a> StructuralObservationSource<'a> {
    pub fn model(self) -> &'a AnalysisModel {
        match self {
            Self::Linear(workload) => &workload.model,
            Self::Frequency(workload) => workload.model(),
            Self::ModalResponseSpectrum(workload) => workload.model(),
            Self::RandomResponse(workload) => workload.model(),
        }
    }

    pub fn procedure_step(self) -> AnalysisExecutionResult<&'a ProcedureStep> {
        match self {
            Self::Linear(workload) => workload.procedure_step(),
            Self::Frequency(workload) => workload.procedure_step(),
            Self::ModalResponseSpectrum(workload) => workload.procedure_step(),
            Self::RandomResponse(workload) => workload.procedure_step(),
        }
    }

    pub const fn observation_execution(self) -> ObservationExecutionPolicy {
        match self {
            Self::Linear(workload) => workload.observation_execution,
            Self::Frequency(workload) => workload.observation_execution(),
            Self::ModalResponseSpectrum(workload) => workload.observation_execution(),
            Self::RandomResponse(workload) => workload.observation_execution(),
        }
    }

    pub const fn formulation_requirement(self) -> FormulationRequirement {
        match self {
            Self::Linear(workload) => workload.formulation_requirement,
            Self::Frequency(workload) => workload.formulation_requirement(),
            Self::ModalResponseSpectrum(workload) => workload.formulation_requirement(),
            Self::RandomResponse(workload) => workload.formulation_requirement(),
        }
    }

    pub fn validate(self) -> AnalysisExecutionResult<()> {
        match self {
            Self::Linear(workload) => workload.validate(),
            Self::Frequency(workload) => workload.validate(),
            Self::ModalResponseSpectrum(workload) => workload.validate(),
            Self::RandomResponse(workload) => workload.validate(),
        }
    }
}
