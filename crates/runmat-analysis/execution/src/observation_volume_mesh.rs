//! Family-neutral admission of observations over native node/element fields recovered on the
//! canonical solver mesh, real or complex. The producing workload is authenticated as
//! provenance only; the plan re-resolves from the authored step and the mesh projection.
use super::*;

pub(super) fn admit_volume_mesh(
    preparation: &ObservationPreparation,
    source_request: &FieldRecoveryRequest,
    source_fields: &ImportedRecoveredFields,
) -> AnalysisBridgeResult<AdmittedObservation> {
    // Also binds StepIdentity(sequence, step) to the request's producer.
    let (workload, mesh) = admit_recovery_source(preparation, source_request, source_fields)?;
    if !matches!(workload, AdmittedObservationWorkload::VolumeMesh) {
        return Err(AnalysisBridgeError::Identity(
            "volume-mesh recovery requires the neutral volume-mesh workload source",
        ));
    }
    let mesh = mesh.ok_or(AnalysisBridgeError::Identity(
        "volume-mesh observation mesh absent",
    ))?;
    let source_result = unique_workload_input(
        &preparation.source_host,
        AnalysisNumericalObjectKind::SolverResult,
    )?;
    let fields = &source_fields.manifest;
    if source_request
        .source
        .solver_result()
        .map(|identity| identity.digest())
        != Some(source_result.logical_identity)
        || !matches!(
            source_request.topology,
            runmat_analysis_fea::execution::FieldTopologyIdentity::VolumeMesh { mesh_digest }
                if mesh_digest == mesh.canonical_digest
        )
        || fields.identity.digest() != source_fields.output.logical_identity
        || fields.recovery_request != source_request.identity()
        || fields.source != source_request.source
        || !preparation.accelerator_requests.is_empty()
    {
        return Err(AnalysisBridgeError::Identity(
            "volume-mesh observation source differs from its original recovery, mesh or serial envelope",
        ));
    }
    let catalog = ObservationFieldCatalog::volume_mesh_recovery(
        source_request.clone(),
        fields.clone(),
        preparation.sequence,
        preparation.step.clone(),
    )?;
    // Resolved from the catalog's own step and manifest on this mesh: canonical by construction.
    let (projection, authority) = catalog.resolve_volume_mesh_plan(&mesh, preparation.limits)?;
    let catalog = authority.catalog();
    let plan = authority.plan().clone();
    let selected = catalog
        .canonical_selected_fields(&plan, ObservationSelectionCoordinate::VolumeMeshRecovery)?;
    let selection = ObservationFieldSelection::volume_mesh_recovery(
        source_request,
        fields,
        projection.projection_digest,
        selected,
    )?;
    catalog.validate_volume_mesh_selection_metadata(&selection, projection.projection_digest)?;
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
        volume_mesh: Some(authority),
    })
}
