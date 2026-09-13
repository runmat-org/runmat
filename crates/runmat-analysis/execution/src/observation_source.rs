use runmat_analysis_core::AnalysisModel;
use runmat_analysis_fea::{
    execution::{
        AnalysisNumericalObjectKind, StructuralFrequencyWorkload, StructuralLinearWorkload,
    },
    results::observation::ObservationExecutionPolicy,
    ComputeBackend,
};

use crate::{AnalysisBridgeError, AnalysisBridgeResult, PreparedAnalysisObjects};

/// Canonical model workload that produced fields consumed by structural observations.
///
/// The variant is semantic: an ordinary structural solve and one independently scheduled
/// frequency entry have different source contracts even though observation evaluation shares the
/// same topology and partition machinery.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum StructuralObservationWorkloadSource {
    Linear(PreparedAnalysisObjects),
    Frequency(PreparedAnalysisObjects),
}

impl StructuralObservationWorkloadSource {
    pub(crate) const fn kind(&self) -> AnalysisNumericalObjectKind {
        match self {
            Self::Linear(_) => AnalysisNumericalObjectKind::SolverInvocation,
            Self::Frequency(_) => AnalysisNumericalObjectKind::StructuralFrequencyWorkload,
        }
    }

    pub(crate) fn objects(&self) -> &PreparedAnalysisObjects {
        match self {
            Self::Linear(objects) | Self::Frequency(objects) => objects,
        }
    }

    pub(crate) fn decode(&self) -> AnalysisBridgeResult<AdmittedStructuralObservationWorkload> {
        let objects = self.objects();
        if objects.manifest.kind != self.kind() {
            return Err(AnalysisBridgeError::Invalid(
                "structural observation workload object has the wrong numerical kind".into(),
            ));
        }
        match self {
            Self::Linear(_) => Ok(AdmittedStructuralObservationWorkload::Linear(
                StructuralLinearWorkload::from_canonical_bytes(&objects.payload())?,
            )),
            Self::Frequency(_) => Ok(AdmittedStructuralObservationWorkload::Frequency(
                StructuralFrequencyWorkload::from_canonical_bytes(&objects.payload())?,
            )),
        }
    }
}

pub(crate) enum AdmittedStructuralObservationWorkload {
    Linear(StructuralLinearWorkload),
    Frequency(StructuralFrequencyWorkload),
}

impl AdmittedStructuralObservationWorkload {
    pub(crate) fn model(&self) -> &AnalysisModel {
        match self {
            Self::Linear(workload) => &workload.model,
            Self::Frequency(workload) => workload.model(),
        }
    }

    pub(crate) const fn backend(&self) -> ComputeBackend {
        match self {
            Self::Linear(workload) => workload.backend,
            Self::Frequency(workload) => workload.backend(),
        }
    }

    pub(crate) const fn observation_execution(&self) -> ObservationExecutionPolicy {
        match self {
            Self::Linear(workload) => workload.observation_execution,
            Self::Frequency(workload) => workload.observation_execution(),
        }
    }
}
