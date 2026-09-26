use runmat_analysis_core::AnalysisModel;
use runmat_analysis_fea::{
    execution::{
        AnalysisNumericalObjectKind, StructuralFrequencyWorkload, StructuralLinearWorkload,
        ThermalWorkload,
    },
    results::observation::ObservationExecutionPolicy,
    ComputeBackend,
};

use crate::{AnalysisBridgeError, AnalysisBridgeResult, PreparedAnalysisObjects};

/// Canonical model workload that produced fields consumed by shared observations.
///
/// The variant is semantic: an ordinary structural solve and one independently scheduled
/// frequency entry and a thermal solve have distinct provenance while sharing partition machinery.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ObservationWorkloadSource {
    Linear(PreparedAnalysisObjects),
    Frequency(PreparedAnalysisObjects),
    Thermal(PreparedAnalysisObjects),
}

impl ObservationWorkloadSource {
    pub(crate) const fn kind(&self) -> AnalysisNumericalObjectKind {
        match self {
            Self::Linear(_) => AnalysisNumericalObjectKind::SolverInvocation,
            Self::Frequency(_) => AnalysisNumericalObjectKind::StructuralFrequencyWorkload,
            Self::Thermal(_) => AnalysisNumericalObjectKind::ThermalWorkload,
        }
    }

    pub(crate) fn objects(&self) -> &PreparedAnalysisObjects {
        match self {
            Self::Linear(objects) | Self::Frequency(objects) | Self::Thermal(objects) => objects,
        }
    }

    pub(crate) fn decode(&self) -> AnalysisBridgeResult<AdmittedObservationWorkload> {
        let objects = self.objects();
        if objects.manifest.kind != self.kind() {
            return Err(AnalysisBridgeError::Invalid(
                "observation workload object has the wrong numerical kind".into(),
            ));
        }
        match self {
            Self::Linear(_) => Ok(AdmittedObservationWorkload::Linear(
                StructuralLinearWorkload::from_canonical_bytes(&objects.payload())?,
            )),
            Self::Frequency(_) => Ok(AdmittedObservationWorkload::Frequency(
                StructuralFrequencyWorkload::from_canonical_bytes(&objects.payload())?,
            )),
            Self::Thermal(_) => Ok(AdmittedObservationWorkload::Thermal(
                ThermalWorkload::from_canonical_bytes(&objects.payload())?,
            )),
        }
    }
}

pub(crate) enum AdmittedObservationWorkload {
    Linear(StructuralLinearWorkload),
    Frequency(StructuralFrequencyWorkload),
    Thermal(ThermalWorkload),
}

impl AdmittedObservationWorkload {
    pub(crate) fn model(&self) -> &AnalysisModel {
        match self {
            Self::Linear(workload) => &workload.model,
            Self::Frequency(workload) => workload.model(),
            Self::Thermal(workload) => &workload.model,
        }
    }

    pub(crate) const fn backend(&self) -> ComputeBackend {
        match self {
            Self::Linear(workload) => workload.backend,
            Self::Frequency(workload) => workload.backend(),
            Self::Thermal(workload) => workload.backend,
        }
    }

    pub(crate) const fn observation_execution(&self) -> ObservationExecutionPolicy {
        match self {
            Self::Linear(workload) => workload.observation_execution,
            Self::Frequency(workload) => workload.observation_execution(),
            // ThermalWorkload has no authored partition override. Use the shared immutable
            // serial policy, also reconstructed by workers; never accept an unbound caller hint.
            Self::Thermal(_) => ObservationExecutionPolicy::serial(),
        }
    }
}
