//! Authenticate the original sampled-field producer before shared partition evaluation.
use super::*;
use runmat_analysis_fea::results::{
    field::{SampledVolumeFieldTopology, VolumeFieldSamplingRule},
    observation::{
        ObservationExecutionPolicy, ObservationFieldCatalog, ObservationFieldCatalogSource,
    },
};

pub(super) fn admit_source(
    objects: &runmat_analysis_execution::PreparedAnalysisObjects,
    catalog: &ObservationFieldCatalog,
    source: &ObservationEvaluationRequest,
    partition: &ObservationPartitionPlan,
    mesh: Option<&SolverMeshArtifact>,
) -> AnalysisBridgeResult<(
    runmat_analysis_core::AnalysisModel,
    SampledVolumeFieldTopology,
)> {
    use runmat_analysis_fea::execution::{MagnetostaticWorkload, MaxwellHarmonicWorkload};
    let (model, producer, workload_identity) = match objects.manifest.kind {
        AnalysisNumericalObjectKind::MagnetostaticWorkload => {
            let workload = MagnetostaticWorkload::from_canonical_bytes(&objects.payload())?;
            let producer = workload.producer_identity()?;
            let identity = workload.identity()?.digest();
            (workload.model, producer, identity)
        }
        AnalysisNumericalObjectKind::MaxwellHarmonicWorkload => {
            let workload = MaxwellHarmonicWorkload::from_canonical_bytes(&objects.payload())?;
            let producer = workload.producer_identity()?;
            let identity = workload.identity()?.digest();
            (workload.model, producer, identity)
        }
        _ => {
            return Err(AnalysisBridgeError::Identity(
                "sampled observation workload kind differs",
            ))
        }
    };
    if objects.manifest.logical_identity != workload_identity {
        return Err(AnalysisBridgeError::Identity(
            "sampled observation original workload identity differs",
        ));
    }
    let mesh = mesh.ok_or(AnalysisBridgeError::Identity(
        "sampled observation mesh absent",
    ))?;
    let ObservationFieldCatalogSource::SampledRecovery {
        sampling,
        recovery_request,
        manifest,
    } = catalog.source()
    else {
        return Err(AnalysisBridgeError::Identity(
            "sampled observation requires sampled recovery catalog",
        ));
    };
    let native_definitions = match objects.manifest.kind {
        AnalysisNumericalObjectKind::MagnetostaticWorkload => {
            runmat_analysis_fea::execution::magnetic_native_field_definitions(recovery_request)?
        }
        AnalysisNumericalObjectKind::MaxwellHarmonicWorkload => {
            runmat_analysis_fea::execution::maxwell_native_field_definitions(recovery_request)?
        }
        _ => {
            return Err(AnalysisBridgeError::Identity(
                "sampled observation workload kind differs",
            ))
        }
    };
    if manifest
        .fields
        .iter()
        .map(|entry| &entry.definition)
        .ne(native_definitions.iter())
    {
        return Err(AnalysisBridgeError::Identity(
            "sampled observation native field definitions differ from typed producer",
        ));
    }
    sampling
        .validate_for_mesh(mesh)
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    if sampling.rule() != VolumeFieldSamplingRule::TetrahedronKeastFifteen {
        return Err(AnalysisBridgeError::Identity(
            "sampled observation sampling rule differs",
        ));
    }
    catalog.validate_sampled_selection(&source.selection)?;
    catalog.validate_sampled_plan(&source.plan, &model.sequence.steps[0], producer, mesh)?;
    let expected = ObservationPartitionPlan::derive(
        &source.plan,
        &partition.observation_id,
        ObservationExecutionPolicy::serial(),
    )
    .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    if partition != &expected {
        return Err(AnalysisBridgeError::Identity(
            "sampled observation partition policy differs from original workload",
        ));
    }
    Ok((model, sampling.as_ref().clone()))
}
