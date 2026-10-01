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
    let workload = runmat_analysis_fea::execution::MagnetostaticWorkload::from_canonical_bytes(
        &objects.payload(),
    )?;
    if objects.manifest.logical_identity != workload.identity()?.digest() {
        return Err(AnalysisBridgeError::Identity(
            "magnetic observation original workload identity differs",
        ));
    }
    let mesh = mesh.ok_or(AnalysisBridgeError::Identity(
        "magnetic observation mesh absent",
    ))?;
    let ObservationFieldCatalogSource::SampledRecovery {
        sampling,
        recovery_request,
        manifest,
    } = catalog.source()
    else {
        return Err(AnalysisBridgeError::Identity(
            "magnetic observation requires sampled recovery catalog",
        ));
    };
    let native_definitions =
        runmat_analysis_fea::execution::magnetic_native_field_definitions(recovery_request)?;
    if manifest
        .fields
        .iter()
        .map(|entry| &entry.definition)
        .ne(native_definitions.iter())
    {
        return Err(AnalysisBridgeError::Identity(
            "magnetic observation native field definitions differ",
        ));
    }
    sampling
        .validate_for_mesh(mesh)
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    if sampling.rule() != VolumeFieldSamplingRule::TetrahedronKeastFifteen {
        return Err(AnalysisBridgeError::Identity(
            "magnetic observation sampling rule differs",
        ));
    }
    catalog.validate_sampled_selection(&source.selection)?;
    catalog.validate_sampled_plan(
        &source.plan,
        &workload.model.sequence.steps[0],
        workload.producer_identity()?,
        mesh,
    )?;
    let expected = ObservationPartitionPlan::derive(
        &source.plan,
        &partition.observation_id,
        ObservationExecutionPolicy::serial(),
    )
    .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    if partition != &expected {
        return Err(AnalysisBridgeError::Identity(
            "magnetic observation partition policy differs from original workload",
        ));
    }
    Ok((workload.model, sampling.as_ref().clone()))
}
