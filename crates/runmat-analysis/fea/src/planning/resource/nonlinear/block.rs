//! Whole numerical envelope plus separately admitted immutable block transport.
use super::*;
use crate::{
    constraints::CanonicalStructuralConstraintPlan,
    execution::{NonlinearSolidBlockArtifact, NonlinearSolidBlockRequest, NonlinearSolidWorkload},
    post::fields::structural_static_observable_field_definitions,
    results::observation::{
        resolve_structural_observation_plan, ObservationExecutionPolicy, ObservationLimits,
    },
    solve::preconditioner::SpdPreconditionerKind,
};
use runmat_execution::Digest;
use runmat_meshing_core::SolverMeshArtifact;

#[cfg(test)]
mod tests;

pub(in crate::planning::resource) fn derive(
    workload: &NonlinearSolidWorkload,
    mesh: &SolverMeshArtifact,
    output_limit: usize,
) -> Result<FeaResourcePlanningInput, ResourcePlanningError> {
    derive_for(
        workload,
        mesh,
        output_limit,
        false,
        LocalExecutionPlan::serial(),
    )
}

pub(in crate::planning::resource) fn derive_recovery(
    workload: &NonlinearSolidWorkload,
    mesh: &SolverMeshArtifact,
    output_limit: usize,
) -> Result<FeaResourcePlanningInput, ResourcePlanningError> {
    derive_for(
        workload,
        mesh,
        output_limit,
        true,
        LocalExecutionPlan::serial(),
    )
}

fn derive_for(
    workload: &NonlinearSolidWorkload,
    mesh: &SolverMeshArtifact,
    output_limit: usize,
    recovery: bool,
    local_execution: LocalExecutionPlan,
) -> Result<FeaResourcePlanningInput, ResourcePlanningError> {
    let invalid = |e: &dyn std::fmt::Display| ResourcePlanningError::Invalid(e.to_string());
    workload.validate().map_err(|e| invalid(&e))?;
    let constraints =
        CanonicalStructuralConstraintPlan::compile_for_model(&workload.model, Some(mesh))
            .map_err(|e| invalid(&e))?;
    let fields = structural_static_observable_field_definitions().map_err(|e| invalid(&e))?;
    let step = &workload.model.sequence.steps[0];
    let observations = resolve_structural_observation_plan(
        step,
        &workload.model,
        Some(mesh),
        &constraints,
        &fields,
        ObservationLimits::default(),
    )
    .map_err(|e| invalid(&e))?;
    derive_resource_input(
        StaticResourcePreflight {
            compilation: None,
            model: &workload.model,
            step,
            backend: crate::ComputeBackend::Cpu,
            preconditioner: SpdPreconditionerKind::Auto,
            observation_execution: ObservationExecutionPolicy::serial(),
            mesh: Some(mesh),
            constraints: &constraints,
            observations: &observations,
            accelerator_requests: &[],
            local_execution,
            prepared_inputs: None,
        },
        &workload.controls,
        output_limit,
        recovery,
    )
}

fn payload_bound(
    input: &FeaResourcePlanningInput,
    request: &NonlinearSolidBlockRequest,
) -> Result<u64, ResourcePlanningError> {
    let FeaResourceDemand::StructuralNonlinearPipeline(demand) = &input.demand else {
        return Err(ResourcePlanningError::Invalid(
            "missing nonlinear authority".into(),
        ));
    };
    // A complete raw result already contains a full point carrier. A second full
    // result envelope conservatively covers the controller's duplicated carrier,
    // schedule and work. Request bytes and fixed artifact keys are separate.
    let request_bytes = request
        .canonical_bytes()
        .map_err(|e| ResourcePlanningError::Invalid(e.to_string()))?;
    demand
        .operator
        .publication
        .result_artifact_bytes
        .0
        .checked_mul(2)
        .and_then(|n| n.checked_add(request_bytes.len() as u64))
        .and_then(|n| n.checked_add(65_536))
        .ok_or(ResourcePlanningError::Overflow("nonlinear block payload"))
}

/// Construction residency of an encoded block artifact beyond its typed contents. The typed
/// history is already charged to the nonlinear pipeline's persistent storage. The direct
/// encoder reserves the output once and holds out-of-order fields in separate buffers, never a
/// value tree: at most the reserved output plus held copies of the same bytes, plus sorted-map
/// metadata and fixed codec state.
pub fn nonlinear_solid_block_output_construction_bytes(
    encoded_bytes: u64,
) -> Result<u64, ResourcePlanningError> {
    encoded_bytes
        .checked_mul(2)
        .and_then(|bytes| {
            bytes.checked_add(
                2 * runmat_canonical_codec::CANONICAL_MATERIALIZATION_FIXED_BYTES_PER_CONTRACT,
            )
        })
        .ok_or(ResourcePlanningError::Overflow("nonlinear block encoding"))
}

pub fn nonlinear_solid_block_payload_bytes_upper_bound(
    workload: &NonlinearSolidWorkload,
    mesh: &SolverMeshArtifact,
    request: &NonlinearSolidBlockRequest,
) -> Result<u64, ResourcePlanningError> {
    request
        .validate()
        .map_err(|e| ResourcePlanningError::Invalid(e.to_string()))?;
    payload_bound(
        &derive(workload, mesh, request.accepted_block_limit)?,
        request,
    )
}

pub fn build_nonlinear_solid_block_resource_plan(
    workload: &NonlinearSolidWorkload,
    mesh: &SolverMeshArtifact,
    request: &NonlinearSolidBlockRequest,
    predecessor: Option<&NonlinearSolidBlockArtifact>,
    prepared: PreparedNonlinearSolidBlockResourceFacts,
) -> Result<FeaResourcePlan, ResourcePlanningError> {
    let invalid = |e: &dyn std::fmt::Display| ResourcePlanningError::Invalid(e.to_string());
    request
        .validate_for(
            workload,
            Digest::from_bytes(*mesh.canonical_digest.bytes()),
            predecessor,
        )
        .map_err(|e| invalid(&e))?;
    let mut input = derive_for(
        workload,
        mesh,
        request.accepted_block_limit,
        false,
        prepared.local_execution,
    )?;
    let bytes = payload_bound(&input, request)?;
    let materialization = nonlinear_solid_block_output_construction_bytes(bytes)?;
    let receipt_bytes = crate::execution::NONLINEAR_SOLID_RECEIPT_MAX_ENCODED_BYTES;
    let receipt_materialization =
        runmat_canonical_codec::projected_canonical_encoding_resident_bytes_estimate(
            receipt_bytes,
            1,
        )
        .ok_or(ResourcePlanningError::Overflow(
            "nonlinear receipt encoding",
        ))?;
    if prepared.inputs.encoded_payload_bytes.0 == 0
        || prepared.inputs.transport_closure_bytes.0 == 0
        || prepared.inputs.decoded_resident_bytes_admission_budget
            < prepared.inputs.encoded_payload_bytes
        || prepared.output.encoded_payload_bytes_upper_bound.0 != bytes
        || prepared.output.transport_closure_bytes_upper_bound.0 < bytes
        || prepared
            .output
            .construction_resident_bytes_admission_budget
            .0
            < materialization
        || prepared.output.construction_resident_bytes_admission_budget
            < prepared.output.transport_closure_bytes_upper_bound
        || prepared.output_set.encoded_bytes_upper_bound.0 == 0
        || prepared.receipt_output.encoded_payload_bytes_upper_bound.0 != receipt_bytes
        || prepared
            .receipt_output
            .transport_closure_bytes_upper_bound
            .0
            < receipt_bytes
        || prepared
            .receipt_output
            .construction_resident_bytes_admission_budget
            .0
            < receipt_materialization
        || prepared
            .receipt_output
            .construction_resident_bytes_admission_budget
            < prepared.receipt_output.transport_closure_bytes_upper_bound
    {
        return Err(invalid(
            &"nonlinear block input/output materialization is underreserved",
        ));
    }
    let FeaResourceDemand::StructuralNonlinearPipeline(demand) = &mut input.demand else {
        return Err(invalid(&"missing nonlinear authority"));
    };
    demand.operator.numerical_scope_digest = request.identity().map_err(|e| invalid(&e))?;
    let transport_bytes = prepared
        .output
        .transport_closure_bytes_upper_bound
        .0
        .checked_add(
            prepared
                .receipt_output
                .transport_closure_bytes_upper_bound
                .0,
        )
        .ok_or(ResourcePlanningError::Overflow(
            "nonlinear block output transport",
        ))?;
    demand.operator.publication.result_artifact_bytes = ByteCount(transport_bytes);
    demand.operator.publication.result_egress_bytes = ByteCount(transport_bytes);
    demand.additional_persistent_bytes.0 = demand
        .additional_persistent_bytes
        .0
        .checked_add(
            prepared
                .output
                .construction_resident_bytes_admission_budget
                .0,
        )
        .ok_or(ResourcePlanningError::Overflow(
            "nonlinear block persistent encoding",
        ))?;
    demand.additional_persistent_bytes.0 = demand
        .additional_persistent_bytes
        .0
        .checked_add(
            prepared
                .receipt_output
                .construction_resident_bytes_admission_budget
                .0,
        )
        .ok_or(ResourcePlanningError::Overflow(
            "nonlinear block receipt construction",
        ))?;
    demand.block_transport = Some(NonlinearSolidBlockTransportDemand {
        request: request.identity().map_err(|e| invalid(&e))?,
        inputs: prepared.inputs,
        output: prepared.output,
        receipt_output: prepared.receipt_output,
    });
    input.output_set = Some(prepared.output_set);
    build_resource_plan(input)
}
