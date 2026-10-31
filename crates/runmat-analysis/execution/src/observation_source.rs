use runmat_analysis_core::AnalysisModel;
use runmat_analysis_fea::{
    execution::{
        AcousticResponseWorkload, AnalysisNumericalObjectKind, MagnetostaticWorkload,
        MaxwellHarmonicWorkload, NonlinearSolidWorkload, StructuralFrequencyWorkload,
        StructuralLinearWorkload, ThermalWorkload,
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
    NonlinearSolid(PreparedAnalysisObjects),
    Frequency(PreparedAnalysisObjects),
    Thermal(PreparedAnalysisObjects),
    Magnetostatic(PreparedAnalysisObjects),
    MaxwellHarmonic(PreparedAnalysisObjects),
    /// The producing workload of native node/element fields on the canonical solver mesh,
    /// of any family. It authenticates provenance only; plans re-resolve from the authored
    /// step and mesh carried by the volume-mesh catalog, never from a decoded workload.
    VolumeMesh(PreparedAnalysisObjects),
}

impl ObservationWorkloadSource {
    pub(crate) const fn kind(&self) -> AnalysisNumericalObjectKind {
        match self {
            Self::Linear(_) => AnalysisNumericalObjectKind::SolverInvocation,
            Self::NonlinearSolid(_) => AnalysisNumericalObjectKind::NonlinearSolidWorkload,
            Self::Frequency(_) => AnalysisNumericalObjectKind::StructuralFrequencyWorkload,
            Self::Thermal(_) => AnalysisNumericalObjectKind::ThermalWorkload,
            Self::Magnetostatic(_) => AnalysisNumericalObjectKind::MagnetostaticWorkload,
            Self::MaxwellHarmonic(_) => AnalysisNumericalObjectKind::MaxwellHarmonicWorkload,
            Self::VolumeMesh(objects) => objects.manifest.kind,
        }
    }

    pub(crate) fn objects(&self) -> &PreparedAnalysisObjects {
        match self {
            Self::Linear(objects)
            | Self::NonlinearSolid(objects)
            | Self::Frequency(objects)
            | Self::Thermal(objects)
            | Self::Magnetostatic(objects)
            | Self::MaxwellHarmonic(objects)
            | Self::VolumeMesh(objects) => objects,
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
            Self::NonlinearSolid(_) => {
                let workload = NonlinearSolidWorkload::from_canonical_bytes(&objects.payload())?;
                if workload.identity()?.digest() != objects.manifest.logical_identity {
                    return Err(AnalysisBridgeError::Identity(
                        "nonlinear observation workload identity differs",
                    ));
                }
                runmat_analysis_fea::execution::StructuralObservationSource::NonlinearSolid(
                    &workload,
                )
                .validate()?;
                Ok(AdmittedObservationWorkload::NonlinearSolid(Box::new(
                    workload,
                )))
            }
            Self::Linear(_) => Ok(AdmittedObservationWorkload::Linear(
                StructuralLinearWorkload::from_canonical_bytes(&objects.payload())?,
            )),
            Self::Frequency(_) => Ok(AdmittedObservationWorkload::Frequency(
                StructuralFrequencyWorkload::from_canonical_bytes(&objects.payload())?,
            )),
            Self::Thermal(_) => Ok(AdmittedObservationWorkload::Thermal(
                ThermalWorkload::from_canonical_bytes(&objects.payload())?,
            )),
            Self::Magnetostatic(_) => Ok(AdmittedObservationWorkload::Magnetostatic(
                MagnetostaticWorkload::from_canonical_bytes(&objects.payload())?,
            )),
            Self::MaxwellHarmonic(_) => Ok(AdmittedObservationWorkload::MaxwellHarmonic(
                MaxwellHarmonicWorkload::from_canonical_bytes(&objects.payload())?,
            )),
            Self::VolumeMesh(_) => {
                // A volume-mesh catalog carries the authored step, but the recovery source's
                // typed workload is still an authenticated input. Decode the acoustic variant
                // before using the family-neutral observation machinery.
                if objects.manifest.kind == AnalysisNumericalObjectKind::AcousticResponseWorkload {
                    let workload =
                        AcousticResponseWorkload::from_canonical_bytes(&objects.payload())?;
                    if workload.identity()?.digest() != objects.manifest.logical_identity {
                        return Err(AnalysisBridgeError::Identity(
                            "acoustic response observation workload identity differs",
                        ));
                    }
                }
                Ok(AdmittedObservationWorkload::VolumeMesh)
            }
        }
    }
}

pub(crate) enum AdmittedObservationWorkload {
    Linear(StructuralLinearWorkload),
    NonlinearSolid(Box<NonlinearSolidWorkload>),
    Frequency(StructuralFrequencyWorkload),
    Thermal(ThermalWorkload),
    Magnetostatic(MagnetostaticWorkload),
    MaxwellHarmonic(MaxwellHarmonicWorkload),
    VolumeMesh,
}

impl AdmittedObservationWorkload {
    pub(crate) fn model(&self) -> AnalysisBridgeResult<&AnalysisModel> {
        Ok(match self {
            Self::Linear(workload) => &workload.model,
            Self::NonlinearSolid(workload) => &workload.model,
            Self::Frequency(workload) => workload.model(),
            Self::Thermal(workload) => &workload.model,
            Self::Magnetostatic(workload) => &workload.model,
            Self::MaxwellHarmonic(workload) => &workload.model,
            Self::VolumeMesh => {
                return Err(AnalysisBridgeError::Identity(
                    "volume-mesh observations resolve from their catalog step, not a model",
                ))
            }
        })
    }

    pub(crate) const fn backend(&self) -> ComputeBackend {
        match self {
            Self::Linear(workload) => workload.backend,
            Self::Frequency(workload) => workload.backend(),
            Self::Thermal(workload) => workload.backend,
            Self::NonlinearSolid(_)
            | Self::Magnetostatic(_)
            | Self::MaxwellHarmonic(_)
            | Self::VolumeMesh => ComputeBackend::Cpu,
        }
    }

    pub(crate) const fn observation_execution(&self) -> ObservationExecutionPolicy {
        match self {
            Self::Linear(workload) => workload.observation_execution,
            Self::Frequency(workload) => workload.observation_execution(),
            // ThermalWorkload has no authored partition override. Use the shared immutable
            // serial policy, also reconstructed by workers; never accept an unbound caller hint.
            Self::NonlinearSolid(_)
            | Self::Thermal(_)
            | Self::Magnetostatic(_)
            | Self::MaxwellHarmonic(_)
            | Self::VolumeMesh => ObservationExecutionPolicy::serial(),
        }
    }
}
