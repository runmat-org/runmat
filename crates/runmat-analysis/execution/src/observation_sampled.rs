//! Source admission for retained sampled native fields; no numerical recovery here.
use super::*;
use runmat_analysis_fea::results::field::{SampledVolumeFieldTopology, VolumeFieldSamplingRule};
use runmat_analysis_fea::results::observation::resolve_sampled_observation_plan;

pub(super) fn admit_sampled(
    preparation: &ObservationPreparation,
    source_request: &FieldRecoveryRequest,
    source_fields: &ImportedRecoveredFields,
    sampling: &SampledVolumeFieldTopology,
) -> AnalysisBridgeResult<AdmittedObservation> {
    let (workload, mesh) = admit_recovery_source(preparation, source_request, source_fields)?;
    let (producer, native_definitions) = match &workload {
        AdmittedObservationWorkload::Magnetostatic(magnetic) => (
            magnetic.producer_identity()?,
            runmat_analysis_fea::execution::magnetic_native_field_definitions(source_request)?,
        ),
        AdmittedObservationWorkload::MaxwellHarmonic(maxwell) => (
            maxwell.producer_identity()?,
            runmat_analysis_fea::execution::maxwell_native_field_definitions(source_request)?,
        ),
        _ => {
            return Err(AnalysisBridgeError::Identity(
                "sampled recovery requires its original supported workload",
            ))
        }
    };
    let mesh = mesh.ok_or(AnalysisBridgeError::Identity(
        "sampled observation mesh absent",
    ))?;
    let source_result = unique_workload_input(
        &preparation.source_host,
        AnalysisNumericalObjectKind::SolverResult,
    )?;
    if source_request
        .source
        .solver_result()
        .map(|identity| identity.digest())
        != Some(source_result.logical_identity)
    {
        return Err(AnalysisBridgeError::Identity(
            "sampled recovery differs from its original retained numerical result",
        ));
    }

    if source_fields
        .manifest
        .fields
        .iter()
        .map(|entry| &entry.definition)
        .ne(native_definitions.iter())
    {
        return Err(AnalysisBridgeError::Identity(
            "sampled observation native field definitions differ",
        ));
    }
    sampling
        .validate_for_mesh(&mesh)
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    if sampling.rule() != VolumeFieldSamplingRule::TetrahedronKeastFifteen
        || producer != source_request.producer
        || !preparation.accelerator_requests.is_empty()
        || source_fields.output.logical_identity != source_fields.manifest.identity.digest()
    {
        return Err(AnalysisBridgeError::Identity(
            "sampled source producer, native layout, output or serial envelope differs",
        ));
    }
    let catalog = ObservationFieldCatalog::sampled_recovery(
        source_request.clone(),
        source_fields.manifest.clone(),
        sampling.clone(),
    )?;
    let definitions = source_fields
        .manifest
        .fields
        .iter()
        .map(|entry| entry.definition.clone())
        .collect::<Vec<_>>();
    let plan = resolve_sampled_observation_plan(
        &preparation.step,
        sampling,
        &mesh,
        &definitions,
        preparation.limits,
    )
    .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    catalog.validate_sampled_plan(&plan, &preparation.step, producer, &mesh)?;
    let selected = catalog
        .canonical_selected_fields(&plan, ObservationSelectionCoordinate::SampledRecovery)?;
    let selection = ObservationFieldSelection::sampled_recovery(
        source_request,
        &source_fields.manifest,
        sampling,
        selected,
    )?;
    catalog.validate_sampled_selection(&selection)?;
    let request = ObservationEvaluationRequest::new(selection.clone(), plan)?;
    let field_catalog = prepare_analysis_objects(
        AnalysisNumericalObjectKind::ObservationFieldCatalog,
        catalog.identity(),
        &catalog.canonical_bytes()?,
        preparation.chunk_bytes,
        ObjectNamespace::InputValue,
        preparation.inventory_limits,
    )?;
    Ok(AdmittedObservation {
        request,
        selection,
        field_catalog,
        workload,
        mesh: Some(mesh),
        accelerator_requests: Vec::new(),
        volume_mesh: None,
    })
}
