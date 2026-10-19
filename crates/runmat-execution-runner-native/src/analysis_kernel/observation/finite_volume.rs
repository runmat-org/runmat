//! Native cell/all-owner-face observations retain the original authored geometry.
use super::*;
use runmat_analysis_fea::{
    execution::FluidSteadyWorkload,
    results::observation::{ObservationExecutionPolicy, ObservationFieldCatalog},
    topology::{AnalysisProjection, FiniteVolumeTopology},
};

pub(super) fn admit_source(
    objects: &runmat_analysis_execution::PreparedAnalysisObjects,
    catalog: &ObservationFieldCatalog,
    source: &ObservationEvaluationRequest,
    partition: &ObservationPartitionPlan,
    mesh: Option<&SolverMeshArtifact>,
) -> AnalysisBridgeResult<(
    runmat_analysis_core::AnalysisModel,
    AnalysisProjection,
    FiniteVolumeTopology,
)> {
    let workload = FluidSteadyWorkload::from_canonical_bytes(&objects.payload())?;
    let mesh = mesh.ok_or(AnalysisBridgeError::Identity(
        "finite-volume observation mesh absent",
    ))?;
    if workload.identity()?.digest() != objects.manifest.logical_identity
        || workload.producer_identity()? != source.producer
    {
        return Err(AnalysisBridgeError::Identity(
            "finite-volume observation authored authority differs",
        ));
    }
    catalog.validate_finite_volume_selection(&source.selection)?;
    catalog.validate_finite_volume_plan(&source.plan, &workload, mesh)?;
    let expected = ObservationPartitionPlan::derive(
        &source.plan,
        &partition.observation_id,
        ObservationExecutionPolicy::serial(),
    )
    .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    if partition != &expected {
        return Err(AnalysisBridgeError::Identity(
            "finite-volume observation requires serial final policy",
        ));
    }
    let projection = AnalysisProjection::derive(mesh)
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let geometry = FiniteVolumeTopology::derive(&projection, workload.geometry_policy)
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    Ok((workload.model, projection, geometry))
}
