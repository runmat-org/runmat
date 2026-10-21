//! Bounded original-native finite-volume observation admission. No re-solve or full field gather.
use super::*;
use runmat_analysis_fea::{
    planning::resource::{FeaResourceDemand, FeaResourcePhase},
    results::{
        field::FiniteVolumeFieldTopology, observation::resolve_finite_volume_observation_plan,
    },
    topology::AnalysisProjection,
};

fn fail(message: &'static str) -> AnalysisBridgeError {
    AnalysisBridgeError::Identity(message)
}

fn bind(
    host: &AnalysisHostWorkload,
    objects: &PreparedAnalysisObjects,
) -> AnalysisBridgeResult<()> {
    let input = unique_workload_input(host, objects.manifest.kind)?;
    if input.logical_identity != objects.manifest.logical_identity
        || input.payload_digest != objects.manifest.payload_digest
        || input.manifest_digest != objects.root.digest
        || input.manifest_encoded_length != objects.root.encoded_length
        || input.reference_kind != runmat_execution::value::ValueRefKind::DriverObject
    {
        return Err(fail(
            "finite-volume observation original input closure differs",
        ));
    }
    Ok(())
}

pub(super) fn admit_finite_volume_observation_source_budget(
    p: &ObservationPreparation,
    request: &FieldRecoveryRequest,
    layout: &FiniteVolumeFieldTopology,
    resource_plan: &PreparedAnalysisObjects,
) -> AnalysisBridgeResult<()> {
    p.source_host.validate()?;
    request.validate()?;
    layout
        .validate()
        .map_err(|e| AnalysisBridgeError::Invalid(e.to_string()))?;
    bind(&p.source_host, resource_plan)?;
    let plan = crate::decode_admitted_resource_plan(resource_plan, p.inventory_limits)?;
    let FeaResourceDemand::FluidFieldRecovery(demand) = &plan.input.demand else {
        return Err(fail(
            "finite-volume observation requires original recovery reservation",
        ));
    };
    let source = unique_workload_input(&p.source_host, AnalysisNumericalObjectKind::SolverResult)?;
    let layout_input = unique_workload_input(
        &p.source_host,
        AnalysisNumericalObjectKind::FieldSourceBinding,
    )?;
    let request_input = unique_workload_input(
        &p.source_host,
        AnalysisNumericalObjectKind::FieldRecoveryRequest,
    )?;
    let layout_digest = runmat_execution::Digest::sha256(
        &layout
            .canonical_bytes()
            .map_err(|e| AnalysisBridgeError::Invalid(e.to_string()))?,
    );
    if plan.input.phase != FeaResourcePhase::FluidFieldRecovery(request.identity())
        || demand.request != request.identity()
        || request.source.solver_result() != Some(demand.source_result)
        || source.logical_identity != demand.source_result.digest()
        || source.reference_kind != runmat_execution::value::ValueRefKind::ResultObject
        || demand.topology_digest != layout.identity().discretization_digest()
        || layout_input.logical_identity != layout_digest
        || layout_input.payload_digest != layout_digest
        || request_input.logical_identity != request.identity().digest()
        || demand.inputs.decoded_resident_bytes_admission_budget.0
            < demand.actual_decode_resident_bytes.0
    {
        return Err(fail(
            "finite-volume observation original request/result/layout reservation differs",
        ));
    }
    let workload = p.source_workload.objects();
    let mesh = p
        .mesh
        .as_ref()
        .ok_or_else(|| fail("finite-volume observation mesh absent"))?;
    let mut encoded = 0_u64;
    for objects in [workload, mesh] {
        runmat_execution_artifact::object::validate_inventory(
            &objects.objects,
            p.inventory_limits,
        )?;
        crate::portable_source_admission::authenticate_source_objects(
            objects,
            objects.manifest.kind,
        )?;
        bind(&p.source_host, objects)?;
        encoded = encoded
            .checked_add(objects.manifest.encoded_length)
            .ok_or_else(|| fail("finite-volume observation wire overflow"))?;
    }
    // Count authenticated borrowed wire before the profiler joins either source payload.
    if encoded > demand.inputs.encoded_payload_bytes.0
        || encoded > demand.inputs.decoded_resident_bytes_admission_budget.0
    {
        return Err(fail(
            "finite-volume observation wire exceeds original recovery admission",
        ));
    }
    let facts =
        crate::object::prepared_input_resource_facts_from_refs([workload, mesh].into_iter())?;
    if facts.decoded_resident_bytes_admission_budget.0
        > demand.inputs.decoded_resident_bytes_admission_budget.0
        || facts.transport_closure_bytes.0 > demand.inputs.transport_closure_bytes.0
    {
        return Err(fail(
            "finite-volume observation actual decode exceeds original recovery admission",
        ));
    }
    Ok(())
}

pub(super) fn admit_finite_volume(
    p: &ObservationPreparation,
    request: &FieldRecoveryRequest,
    fields: &ImportedRecoveredFields,
    layout: &FiniteVolumeFieldTopology,
    resource_plan: &PreparedAnalysisObjects,
) -> AnalysisBridgeResult<AdmittedObservation> {
    if !matches!(p.source_workload, ObservationWorkloadSource::FluidSteady(_)) {
        return Err(fail(
            "finite-volume observations require original fluid workload",
        ));
    }
    admit_finite_volume_observation_source_budget(p, request, layout, resource_plan)?;
    let (workload, mesh) = admit_recovery_source(p, request, fields)?;
    let AdmittedObservationWorkload::FluidSteady(fluid) = &workload else {
        unreachable!()
    };
    fluid.validate()?;
    let mesh = mesh.ok_or_else(|| fail("finite-volume observation mesh absent"))?;
    let definitions =
        runmat_analysis_fea::execution::fluid_field_definition_inventory(fluid.coupling())?;
    if request.producer != fluid.producer_identity()?
        || fluid.model.sequence.steps.as_slice() != [p.step.clone()]
        || fields
            .manifest
            .fields
            .iter()
            .map(|entry| &entry.definition)
            .ne(definitions.iter())
        || fields.output.logical_identity != fields.manifest.identity.digest()
        || !p.accelerator_requests.is_empty()
    {
        return Err(fail(
            "finite-volume observation original authored plan/native inventory differs",
        ));
    }
    let projection = AnalysisProjection::derive(&mesh)
        .map_err(|e| AnalysisBridgeError::Invalid(e.to_string()))?;
    let geometry = fluid.fluid_cell_complex(&mesh, &projection)?;
    layout
        .validate_for(&projection, &geometry)
        .map_err(|e| AnalysisBridgeError::Invalid(e.to_string()))?;
    let plan = resolve_finite_volume_observation_plan(
        &p.step,
        layout,
        &projection,
        &geometry,
        &definitions,
        p.limits,
    )
    .map_err(|e| AnalysisBridgeError::Invalid(e.to_string()))?;
    let catalog = ObservationFieldCatalog::finite_volume_recovery(
        request.clone(),
        fields.manifest.clone(),
        layout.clone(),
    )?;
    catalog.validate_finite_volume_plan(&plan, fluid, &mesh)?;
    let selected = catalog
        .canonical_selected_fields(&plan, ObservationSelectionCoordinate::FiniteVolumeRecovery)?;
    let selection = ObservationFieldSelection::finite_volume_recovery(
        request,
        &fields.manifest,
        layout,
        selected,
    )?;
    catalog.validate_finite_volume_selection(&selection)?;
    let request = ObservationEvaluationRequest::new(selection.clone(), plan)?;
    let field_catalog = prepare_analysis_objects(
        AnalysisNumericalObjectKind::ObservationFieldCatalog,
        catalog.identity(),
        &catalog.canonical_bytes()?,
        p.chunk_bytes,
        ObjectNamespace::InputValue,
        p.inventory_limits,
    )?;
    Ok(AdmittedObservation {
        request,
        selection,
        field_catalog,
        workload,
        mesh: Some(mesh),
        accelerator_requests: Vec::new(),
    })
}
