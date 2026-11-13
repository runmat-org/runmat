//! Admission of observations over one accepted sample of an already published thermal solve.
use super::*;
use runmat_analysis_fea::results::observation::resolve_observation_plan;
use runmat_analysis_fea::topology::AnalysisProjection;

pub(super) fn admit_thermal(
    preparation: &ObservationPreparation,
) -> AnalysisBridgeResult<AdmittedObservation> {
    let ObservationPublishedSource::Thermal {
        manifest,
        output,
        access,
        sample_ordinal,
    } = &preparation.source
    else {
        return Err(AnalysisBridgeError::Identity(
            "thermal observation requires thermal source",
        ));
    };
    preparation.source_host.validate()?;
    manifest.validate()?;
    if preparation.source_host.workload.kind != AnalysisWorkKind::SolveThermal
        || preparation.source_host.workload.outputs
            != vec![
                AnalysisNumericalObjectKind::SolverResult,
                AnalysisNumericalObjectKind::NumericalReceipt,
                AnalysisNumericalObjectKind::StatePublicationProfiles,
            ]
        || output.kind != AnalysisNumericalObjectKind::SolverResult
        || output.logical_identity != manifest.identity().digest()
        || preparation.source_host.workload.step != manifest.producer
        || access != &preparation.source_host.artifact_access
        || preparation.chunk_bytes == 0
    {
        return Err(AnalysisBridgeError::Identity(
            "thermal observation source differs from completed solve",
        ));
    }
    let workload = preparation.source_workload.decode()?;
    let AdmittedObservationWorkload::Thermal(thermal) = &workload else {
        return Err(AnalysisBridgeError::Identity(
            "thermal observation requires thermal workload",
        ));
    };
    let workload_objects = preparation.source_workload.objects();
    let workload_input = unique_workload_input(
        &preparation.source_host,
        AnalysisNumericalObjectKind::ThermalWorkload,
    )?;
    if workload_input.logical_identity != thermal.identity()?.digest()
        || workload_input.logical_identity != workload_objects.manifest.logical_identity
        || workload_input.payload_digest != workload_objects.manifest.payload_digest
        || workload_input.manifest_digest != workload_objects.root.digest
    {
        return Err(AnalysisBridgeError::Identity(
            "thermal observation workload differs from producing task",
        ));
    }
    let mesh_objects = preparation
        .mesh
        .as_ref()
        .ok_or(AnalysisBridgeError::Identity(
            "thermal observations require canonical mesh",
        ))?;
    let mesh_input =
        unique_workload_input(&preparation.source_host, AnalysisNumericalObjectKind::Mesh)?;
    if mesh_objects.manifest.kind != AnalysisNumericalObjectKind::Mesh
        || mesh_input.logical_identity != mesh_objects.manifest.logical_identity
        || mesh_input.payload_digest != mesh_objects.manifest.payload_digest
        || mesh_input.manifest_digest != mesh_objects.root.digest
    {
        return Err(AnalysisBridgeError::Identity(
            "thermal observation mesh differs from producing task",
        ));
    }
    let mesh = SolverMeshArtifact::canonical_decode(&mesh_objects.payload())
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let projection = AnalysisProjection::derive(&mesh)
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    if StepIdentity::from_solver_contract(preparation.sequence, &preparation.step)?
        != manifest.producer
        || !thermal
            .model
            .sequence
            .steps
            .iter()
            .any(|step| step == &preparation.step)
    {
        return Err(AnalysisBridgeError::Identity(
            "thermal observation step differs from authored source",
        ));
    }
    let sample_fields =
        manifest
            .fields
            .get(*sample_ordinal)
            .ok_or(AnalysisBridgeError::Identity(
                "thermal observation sample absent",
            ))?;
    let definitions = sample_fields
        .iter()
        .map(|entry| entry.definition.clone())
        .collect::<Vec<_>>();
    let plan = resolve_observation_plan(
        &preparation.step,
        &projection,
        &definitions,
        preparation.limits,
    )
    .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let catalog = ObservationFieldCatalog::thermal(manifest.clone())?;
    let fields = catalog.canonical_selected_fields(
        &plan,
        ObservationSelectionCoordinate::ThermalSample {
            ordinal: *sample_ordinal,
        },
    )?;
    let selection =
        ObservationFieldSelection::thermal(thermal, manifest, &mesh, *sample_ordinal, fields)?;
    catalog.validate_thermal_selection(&selection, thermal, &mesh)?;
    let request = ObservationEvaluationRequest::new(selection.clone(), plan)?;
    let field_catalog = prepare_analysis_objects(
        AnalysisNumericalObjectKind::ObservationFieldCatalog,
        catalog.identity(),
        &catalog.canonical_bytes()?,
        preparation.chunk_bytes,
        ObjectNamespace::InputValue,
        preparation.inventory_limits,
    )?;
    for accelerator in &preparation.accelerator_requests {
        accelerator
            .validate()
            .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    }
    Ok(AdmittedObservation {
        request,
        selection,
        field_catalog,
        workload,
        mesh: Some(mesh),
        accelerator_requests: preparation.accelerator_requests.clone(),
        volume_mesh: None,
    })
}
