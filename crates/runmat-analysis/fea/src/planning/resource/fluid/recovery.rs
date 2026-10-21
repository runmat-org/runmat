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
    // The original native plan names its coupled fields: conjugate studies add temperature.
    let coupling = if request
        .plan
        .requirements
        .iter()
        .any(|r| r.quantity == runmat_analysis_core::ObservationQuantity::Temperature)
    {
        crate::execution::FluidCoupling::ConjugateHeat
    } else {
        crate::execution::FluidCoupling::Standalone
    };
    let definitions =
        crate::execution::fluid_field_definition_inventory(coupling).map_err(invalid)?;
    request
        .plan
        .validate_definitions(definitions.iter().cloned())
        .map_err(invalid)?;
    if !matches!(
        request.topology,
        crate::execution::FieldTopologyIdentity::FiniteVolumeMesh { .. }
    ) {
        return Err(invalid(
            "fluid native recovery requires original all-face finite-volume layout",
        ));
    }
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
    result.validate_for(workload, mesh).map_err(invalid)?;
    let topology = result
        .native_field_topology(workload, mesh)
        .map_err(invalid)?;
    let expected = result
        .native_field_request(workload, mesh)
        .map_err(invalid)?;
    if request != &expected {
        return Err(invalid(
            "fluid native recovery differs from original retained source/layout/request",
        ));
    }
    let projection = derive_fluid_steady_resource_projection(workload, mesh)?;
    let native = result
        .native_field_resident_bytes_estimate(workload, mesh)
        .map_err(invalid)?;
    let (partitions, partition_workspace) = fluid_field_recovery_shape(request)?;
    let output_bytes = partitions.into_iter().try_fold(0, admission::add)?;
    let profiles = [
        profile_serialized(workload, LIMITS).map_err(invalid)?,
        profile_serialized(mesh, LIMITS).map_err(invalid)?,
        profile_serialized(result, LIMITS).map_err(invalid)?,
        profile_serialized(request, LIMITS).map_err(invalid)?,
        profile_serialized(&topology, LIMITS).map_err(invalid)?,
    ];
    let wire_workspace = profiles.into_iter().try_fold(0, |sum, p| {
        admission::add(
            sum,
            admission::add(
                p.encoded_bytes_upper_bound,
                p.materialization_resident_bytes_estimate,
            )?,
        )
    })?;
    let construction = admission::add(native, partition_workspace.0)?;
    let validation = admission::add(
        projection.phase_resident_bytes_upper_bound.0,
        wire_workspace,
    )?;
    admission::require_ceiling(
        admission::add(construction, validation)?,
        workload.resident_ceiling().map_err(invalid)?,
    )?;
    // Complete input wire profiles precede materialization of these five canonical buffers.
    let bytes = [
        workload.canonical_bytes().map_err(invalid)?,
        mesh.canonical_encode().map_err(invalid)?,
        result.canonical_bytes().map_err(invalid)?,
        request.canonical_bytes().map_err(invalid)?,
        topology.canonical_bytes().map_err(invalid)?,
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
                runmat_canonical_codec::canonical_payload_materialization_resident_bytes_estimate(
                    b,
                )
                .map_err(invalid)?,
            )?,
        ))
    })?;
    if facts.inputs.encoded_payload_bytes.0 != encoded
        || facts.inputs.transport_closure_bytes.0 < encoded
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
        worker: governed::worker(workload),
        accuracy_ceiling: governed::ceiling(workload)?,
    })
}
