use super::admission;
use super::*;
use crate::{
    execution::{FieldRecoveryRequest, FluidSteadySolverResult},
    planning::resource::*,
    results::field::{
        RecoveredField, RecoveredFieldArtifactManifest, RecoveredFieldPartition,
        RecoveredFieldSample, RECOVERED_FIELD_PARTITION_ENTITY_COUNT,
    },
};

/// Catalog plus exact native final-value partitions; no numerical arrays are allocated.
pub fn fluid_field_recovery_shape(
    request: &FieldRecoveryRequest,
) -> Result<(Vec<u64>, ByteCount), ResourcePlanningError> {
    request.validate().map_err(invalid)?;
    // The request's requirements name the coupled fields and the fluid support.
    let definitions = crate::execution::fluid_native_field_definitions(request).map_err(invalid)?;
    let counts = definitions.iter().map(|definition| {
        let requirement = request.plan.requirements.iter().find(|r| r.quantity == definition.quantity()
            && r.entity_domain == definition.entity_domain()).ok_or_else(|| invalid("fluid native quantity absent"))?;
        let components = definition.storage_component_count().map_err(invalid)? as u64;
        if requirement.recovery != runmat_analysis_core::RecoveryPolicy::Native
            || requirement.source_location != definition.location()
            || requirement.output_location != definition.location()
            || requirement.cost.source_entity_count != requirement.cost.output_entity_count
            || requirement.cost.topology_incidence_count != 0
            || requirement.cost.arithmetic_operation_count != requirement.cost.output_entity_count.checked_mul(components)
                .ok_or(ResourcePlanningError::Overflow("fluid native copy work"))? {
            return Err(invalid("fluid recovery requires exact native copies, not interpolation or synthetic work"));
        }
        usize::try_from(requirement.cost.output_entity_count).map_err(|_| ResourcePlanningError::Overflow("fluid native entities"))
    }).collect::<Result<Vec<_>, _>>()?;
    let catalog =
        RecoveredFieldArtifactManifest::profile_final_shape(request, &definitions, &counts)
            .map_err(invalid)?;
    let mut payloads = vec![catalog.encoded_bytes_upper_bound];
    let mut workspace = admission::add(
        catalog.encoded_bytes_upper_bound,
        catalog.materialization_resident_bytes_estimate,
    )?;
    for (definition, count) in definitions.iter().zip(counts) {
        let identity =
            RecoveredField::identity_profile_shape(definition, RecoveredFieldSample::Final, count)
                .map_err(invalid)?;
        workspace = admission::add(
            workspace,
            admission::add(
                identity.encoded_bytes_upper_bound,
                identity.materialization_resident_bytes_estimate,
            )?,
        )?;
        let mut remaining = count;
        while remaining > 0 {
            let n = remaining.min(RECOVERED_FIELD_PARTITION_ENTITY_COUNT);
            let profile =
                RecoveredFieldPartition::profile_shape(definition, RecoveredFieldSample::Final, n)
                    .map_err(invalid)?;
            payloads.push(profile.encoded_bytes_upper_bound);
            workspace = admission::add(
                workspace,
                admission::add(
                    profile.encoded_bytes_upper_bound,
                    profile.materialization_resident_bytes_estimate,
                )?,
            )?;
            remaining -= n;
        }
    }
    Ok((payloads, ByteCount(workspace)))
}

pub fn build_fluid_field_recovery_resource_plan(
    workload: &FluidSteadyWorkload,
    mesh: &SolverMeshArtifact,
    result: &FluidSteadySolverResult,
    request: &FieldRecoveryRequest,
    facts: PreparedPhaseResourceFacts,
) -> Result<FeaResourcePlan, ResourcePlanningError> {
    // Validates the result against its source, prepared once for the checks below.
    let on_source = result.on_source(workload, mesh).map_err(invalid)?;
    build_fluid_field_recovery_resource_plan_on(&on_source, mesh, request, facts)
}

/// [`build_fluid_field_recovery_resource_plan`] for a result already validated against its
/// source on `mesh`.
pub fn build_fluid_field_recovery_resource_plan_on(
    on_source: &crate::execution::FluidResultOnSource<'_>,
    mesh: &SolverMeshArtifact,
    request: &FieldRecoveryRequest,
    facts: PreparedPhaseResourceFacts,
) -> Result<FeaResourcePlan, ResourcePlanningError> {
    let (workload, result) = (on_source.workload(), on_source.result());
    let expected = on_source.native_field_request().map_err(invalid)?;
    if request != &expected {
        return Err(invalid(
            "fluid native recovery differs from original retained source/layout/request",
        ));
    }
    let projection = derive_fluid_steady_resource_projection(workload, mesh)?;
    let native = on_source
        .native_field_resident_bytes_estimate()
        .map_err(invalid)?;
    let (partitions, partition_workspace) = fluid_field_recovery_shape(request)?;
    let output_bytes = partitions.into_iter().try_fold(0, admission::add)?;
    // The source verification phase bound charges the workload and mesh wire profiles and
    // the result's encoding and materialization alongside source preparation and the
    // numerical verification reserve (validation re-assembles and re-evaluates the retained
    // residual but runs no linear solve). Only the recovery request's wire is new here.
    let request_profile = profile_serialized(request, LIMITS).map_err(invalid)?;
    let request_wire = admission::add(
        request_profile.encoded_bytes_upper_bound,
        request_profile.materialization_resident_bytes_estimate,
    )?;
    let construction = admission::add(native, partition_workspace.0)?;
    let validation = admission::add(
        projection.verification_phase_resident_bytes_upper_bound.0,
        request_wire,
    )?;
    tracing::info!(
        cells = projection.numerical.cells,
        native_bytes = native,
        partition_workspace_bytes = partition_workspace.0,
        source_phase_bytes = projection.phase_resident_bytes_upper_bound.0,
        verification_phase_bytes = projection.verification_phase_resident_bytes_upper_bound.0,
        numerical_bytes = projection.numerical.resident_bytes_upper_bound,
        numerical_verification_bytes = projection.numerical.verification_resident_bytes_upper_bound,
        source_preparation_bytes = projection.source_preparation_resident_bytes.0,
        result_materialization_bytes = projection.result_materialization_resident_bytes.0,
        request_wire_bytes = request_wire,
        "fluid field recovery workspace plan"
    );
    admission::require_ceiling(
        admission::add(construction, validation)?,
        workload.resident_ceiling().map_err(invalid)?,
    )?;
    // Complete input wire profiles precede materialization of these four canonical buffers.
    let bytes = [
        workload.canonical_bytes().map_err(invalid)?,
        mesh.canonical_encode().map_err(invalid)?,
        result.canonical_bytes().map_err(invalid)?,
        request.canonical_bytes().map_err(invalid)?,
    ];
    let (encoded, decoded) = bytes.iter().try_fold((0, 0), |(e, d), b| {
        Ok::<_, ResourcePlanningError>((
            admission::add(
                e,
                u64::try_from(b.len())
                    .map_err(|_| ResourcePlanningError::Overflow("fluid recovery wire"))?,
            )?,
            admission::add(
                d,
                runmat_canonical_codec::canonical_payload_typed_resident_bytes_estimate(b)
                    .map_err(invalid)?,
            )?,
        ))
    })?;
    if facts.inputs.encoded_payload_bytes.0 != encoded
        // Content-addressed repeated chunks can make unique transport smaller than the payload.
        || facts.inputs.transport_closure_bytes.0 == 0
        || facts.inputs.decoded_resident_bytes_admission_budget.0 < decoded
        || facts.output.encoded_payload_bytes_upper_bound.0 != output_bytes
        || facts.output.transport_closure_bytes_upper_bound.0 < output_bytes
        || facts.output.construction_resident_bytes_admission_budget
            < facts.output.transport_closure_bytes_upper_bound
    {
        return Err(invalid(
            "fluid native recovery original input/output closure differs",
        ));
    }
    build_resource_plan(FeaResourcePlanningInput {
        estimation_model_revision: RESOURCE_ESTIMATION_MODEL_REVISION,
        phase: FeaResourcePhase::FluidFieldRecovery(request.identity()),
        demand: FeaResourceDemand::FluidFieldRecovery(FluidFieldRecoveryResourceDemand {
            request: request.identity(),
            source_result: result.identity().map_err(invalid)?,
            topology_digest: request.topology.discretization_digest(),
            construction_workspace_bytes: ByteCount(construction),
            source_validation_workspace_bytes: ByteCount(validation),
            actual_decode_resident_bytes: ByteCount(decoded),
            inputs: facts.inputs,
            output: facts.output,
        }),
        output_set: Some(facts.output_set),
        // Recovery publishes retained solve arrays on one thread; it performs no numerical solve.
        worker: governed::worker(workload, LocalExecutionPlan::serial())?,
        accuracy_ceiling: governed::ceiling(workload)?,
    })
}
