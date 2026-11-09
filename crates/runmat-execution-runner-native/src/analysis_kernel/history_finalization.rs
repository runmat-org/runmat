//! Admitted finalization consumes a checkpoint; it never reruns the producing solve.
use super::*;
use runmat_analysis_fea::execution::{
    ObservationHistoryFinalizationRequest, ObservationHistorySourceContext,
    StructuralObservationSource, ThermalWorkload,
};
use runmat_analysis_fea::planning::resource::{
    build_observation_history_finalization_resource_plan,
    observation_history_finalization_payload_bytes_upper_bound,
    observation_history_series_reduction_payload_bytes_upper_bound,
};
use runmat_analysis_fea::results::observation::{
    ObservationFieldCatalog, ObservationFieldCatalogSource,
};

pub(super) fn execute(
    invocation: AnalysisKernelInvocation<'_>,
) -> AnalysisBridgeResult<Vec<AnalysisKernelOutput>> {
    let request_objects = unique_input(
        &invocation,
        AnalysisNumericalObjectKind::ObservationHistoryFinalizationRequest,
    )?;
    let checkpoint_objects = unique_input(
        &invocation,
        AnalysisNumericalObjectKind::ObservationHistoryCheckpoint,
    )?;
    let catalog_objects = unique_input(
        &invocation,
        AnalysisNumericalObjectKind::ObservationFieldCatalog,
    )?;
    let resource_objects = unique_input(&invocation, AnalysisNumericalObjectKind::ResourcePlan)?;
    let thermal_objects =
        optional_unique_input(&invocation, AnalysisNumericalObjectKind::ThermalWorkload)?;
    let mesh_objects = optional_unique_input(&invocation, AnalysisNumericalObjectKind::Mesh)?;
    let linear_objects =
        optional_unique_input(&invocation, AnalysisNumericalObjectKind::SolverInvocation)?;
    let frequency_objects = optional_unique_input(
        &invocation,
        AnalysisNumericalObjectKind::StructuralFrequencyWorkload,
    )?;
    if invocation.inputs.len()
        != 4 + usize::from(thermal_objects.is_some())
            + usize::from(mesh_objects.is_some())
            + usize::from(linear_objects.is_some())
            + usize::from(frequency_objects.is_some())
        || usize::from(thermal_objects.is_some())
            + usize::from(linear_objects.is_some())
            + usize::from(frequency_objects.is_some())
            > 1
    {
        return Err(AnalysisBridgeError::Identity(
            "history finalization has unexpected inputs",
        ));
    }
    let request =
        ObservationHistoryFinalizationRequest::from_canonical_bytes(&request_objects.payload())?;
    let checkpoint = decode_prepared_observation_history_checkpoint(checkpoint_objects)?;
    let catalog = ObservationFieldCatalog::from_canonical_bytes(&catalog_objects.payload())?;
    if request.identity().digest() != request_objects.manifest.logical_identity
        || checkpoint.manifest().identity.digest() != checkpoint_objects.manifest.logical_identity
        || catalog.identity() != catalog_objects.manifest.logical_identity
        || request.producer != invocation.host.workload.step
    {
        return Err(AnalysisBridgeError::Identity(
            "history finalization input identities differ",
        ));
    }
    let thermal = thermal_objects
        .map(|objects| {
            let workload = ThermalWorkload::from_canonical_bytes(&objects.payload())?;
            if workload.identity()?.digest() != objects.manifest.logical_identity {
                return Err(AnalysisBridgeError::Identity(
                    "history thermal workload identity differs",
                ));
            }
            Ok(workload)
        })
        .transpose()?;
    let mesh = mesh_objects
        .map(|objects| {
            let mesh =
                runmat_meshing_core::SolverMeshArtifact::canonical_decode(&objects.payload())
                    .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
            if runmat_execution::Digest::from_bytes(*mesh.canonical_digest.bytes())
                != objects.manifest.logical_identity
            {
                return Err(AnalysisBridgeError::Identity(
                    "history mesh identity differs",
                ));
            }
            Ok(mesh)
        })
        .transpose()?;
    let linear = linear_objects
        .map(|objects| {
            runmat_analysis_fea::execution::StructuralLinearWorkload::from_canonical_bytes(
                &objects.payload(),
            )
        })
        .transpose()?;
    let frequency = if let Some(objects) = frequency_objects {
        let workload =
            runmat_analysis_fea::execution::StructuralFrequencyWorkload::from_canonical_bytes(
                &objects.payload(),
            )?;
        if workload.identity().digest() != objects.manifest.logical_identity {
            return Err(AnalysisBridgeError::Identity(
                "history frequency workload identity differs",
            ));
        }
        Some(workload)
    } else {
        None
    };
    let structural_source = linear
        .as_ref()
        .map(StructuralObservationSource::Linear)
        .or_else(|| {
            frequency
                .as_ref()
                .map(StructuralObservationSource::Frequency)
        });
    let context = match (
        catalog.source(),
        thermal.as_ref(),
        structural_source,
        mesh.as_ref(),
    ) {
        (ObservationFieldCatalogSource::Thermal { .. }, Some(workload), None, Some(mesh)) => {
            ObservationHistorySourceContext::Thermal { workload, mesh }
        }
        (ObservationFieldCatalogSource::Structural { .. }, None, Some(source), mesh) => {
            ObservationHistorySourceContext::Structural { source, mesh }
        }
        (
            ObservationFieldCatalogSource::EntryHistory { .. },
            None,
            Some(source @ StructuralObservationSource::Frequency(_)),
            mesh,
        ) => ObservationHistorySourceContext::Structural { source, mesh },
        (
            ObservationFieldCatalogSource::VolumeMeshRecovery { .. }
            | ObservationFieldCatalogSource::EntryHistory { .. },
            None,
            None,
            Some(mesh),
        ) => {
            ObservationHistorySourceContext::VolumeMesh { mesh }
        }
        _ => {
            return Err(AnalysisBridgeError::Identity(
                "history source context differs from catalog",
            ))
        }
    };
    request.validate_inputs(&checkpoint, &catalog, context)?;
    let mut expected_outputs = vec![AnalysisNumericalObjectKind::History];
    if request.series_reduction.is_some() {
        expected_outputs.push(AnalysisNumericalObjectKind::SeriesReductionResult);
    }
    if invocation.host.workload.outputs != expected_outputs {
        return Err(AnalysisBridgeError::Identity(
            "history finalization output contract differs",
        ));
    }
    let resource = FeaResourcePlan::from_canonical_bytes(&resource_objects.payload())
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let facts = prepared_input_resource_facts(&inputs_except_resource_plan(&invocation))?;
    let history_bound =
        observation_history_finalization_payload_bytes_upper_bound(&request, &checkpoint)
            .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let series_bound =
        observation_history_series_reduction_payload_bytes_upper_bound(&request, &checkpoint)
            .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let expected = build_observation_history_finalization_resource_plan(
        &request,
        &checkpoint,
        &catalog,
        context,
        runmat_analysis_fea::planning::resource::PreparedPhaseResourceFacts {
            inputs: facts,
            output: projected_output_resource_facts(history_bound)?,
            output_set: projected_output_set_resource_facts(),
        },
        series_bound
            .map(projected_output_resource_facts)
            .transpose()?,
    )
    .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    if resource != expected
        || resource.identity.digest() != resource_objects.manifest.logical_identity
        || !resource.request.accelerators.is_empty()
    {
        return Err(AnalysisBridgeError::Identity(
            "history finalization resource plan differs from authoritative inputs",
        ));
    }
    let finalized = request.finalize(checkpoint, &catalog, context)?;
    let error = |error: runmat_analysis_fea::results::observation::ObservationError| {
        AnalysisBridgeError::Invalid(error.to_string())
    };
    let history_output = match finalized.history.manifest().partial_access {
        runmat_analysis_core::PartialAccessPolicy::WholeArtifact => AnalysisKernelOutput::whole(
            AnalysisNumericalObjectKind::History,
            finalized.history.manifest().identity.digest(),
            finalized.history.canonical_bytes().map_err(error)?,
        ),
        runmat_analysis_core::PartialAccessPolicy::ManifestAndChunks => {
            let mut partitions = Vec::with_capacity(finalized.history.samples().len() + 1);
            partitions.push(AnalysisKernelOutputPartition {
                logical_identity: finalized.history.manifest().identity.digest(),
                payload: finalized
                    .history
                    .manifest()
                    .canonical_bytes()
                    .map_err(error)?,
            });
            for sample in finalized.history.samples() {
                partitions.push(AnalysisKernelOutputPartition {
                    logical_identity: sample.identity.digest(),
                    payload: sample.canonical_bytes().map_err(error)?,
                });
            }
            AnalysisKernelOutput::partitioned(
                AnalysisNumericalObjectKind::History,
                finalized.history.manifest().identity.digest(),
                partitions,
            )
        }
    };
    let mut outputs = vec![history_output];
    if let Some(series) = finalized.series_reduction {
        outputs.push(AnalysisKernelOutput::whole(
            AnalysisNumericalObjectKind::SeriesReductionResult,
            series.identity.digest(),
            series.canonical_bytes().map_err(error)?,
        ));
    }
    Ok(outputs)
}
