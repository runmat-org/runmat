//! Original transient checkpoint phase, using shared host/transport accounting.
use super::*;
use crate::execution::{AnalysisWorkloadIdentity, FluidTransientBlockWorkload};
use runmat_execution::Digest;
use runmat_meshing_core::{CanonicalMeshingContract, SolverMeshArtifact};
use serde::{Deserialize, Serialize};
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FluidTransientBlockResourceProjection {
    pub workload: AnalysisWorkloadIdentity,
    pub topology_digest: Digest,
    pub predecessor: Digest,
    pub start_interval: usize,
    pub end_interval: usize,
    pub charged_operator_work: u64,
    pub phase_resident_bytes: ByteCount,
    pub canonical_input_bytes: ByteCount,
    pub actual_decode_resident_bytes: ByteCount,
    pub checkpoint_encoded_bytes: ByteCount,
    pub checkpoint_resident_bytes: ByteCount,
}
fn invalid(e: impl std::fmt::Display) -> ResourcePlanningError {
    ResourcePlanningError::Invalid(e.to_string())
}
fn add(a: u64, b: u64) -> Result<u64, ResourcePlanningError> {
    a.checked_add(b)
        .ok_or(ResourcePlanningError::Overflow("fluid transient bytes"))
}
pub fn derive_fluid_transient_block_resource_projection(
    request: &FluidTransientBlockWorkload,
    mesh: &SolverMeshArtifact,
) -> Result<FluidTransientBlockResourceProjection, ResourcePlanningError> {
    let prepared = request.prepare(mesh).map_err(invalid)?;
    projection_from_prepared(request, mesh, &prepared)
}
pub(crate) fn projection_from_prepared(
    request: &FluidTransientBlockWorkload,
    mesh: &SolverMeshArtifact,
    prepared: &crate::execution::PreparedFluidTransientContinuation<'_>,
) -> Result<FluidTransientBlockResourceProjection, ResourcePlanningError> {
    let output = prepared.result_encoding_profile().map_err(invalid)?;
    let request_profile = request.encoding_profile().map_err(invalid)?;
    let mesh_profile = runmat_canonical_codec::profile_serialized(
        mesh,
        runmat_canonical_codec::CanonicalLimits::new(usize::MAX, usize::MAX, usize::MAX, 64),
    )
    .map_err(invalid)?;
    let resident = add(
        prepared
            .phase_resident_bytes_upper_bound()
            .map_err(invalid)?,
        add(
            add(
                request_profile.materialization_resident_bytes_estimate,
                mesh_profile.materialization_resident_bytes_estimate,
            )?,
            add(
                add(
                    request_profile.encoded_bytes_upper_bound,
                    mesh_profile.encoded_bytes_upper_bound,
                )?,
                output.encoded_bytes_upper_bound,
            )?,
        )?,
    )?;
    if resident > request.workload.resident_ceiling().map_err(invalid)? {
        return Err(ResourcePlanningError::AccuracyCeiling {
            resource: "fluid transient complete phase heap",
            required: resident,
            maximum: request.workload.resident_ceiling().map_err(invalid)?,
        });
    }
    // Original wire allocation follows complete borrowed-source admission.
    let request_bytes = request.canonical_bytes().map_err(invalid)?;
    let mesh_bytes = mesh.canonical_encode().map_err(invalid)?;
    let decode = add(
        runmat_canonical_codec::canonical_payload_typed_resident_bytes_estimate(&request_bytes)
            .map_err(invalid)?,
        runmat_canonical_codec::canonical_payload_typed_resident_bytes_estimate(&mesh_bytes)
            .map_err(invalid)?,
    )?;
    Ok(FluidTransientBlockResourceProjection {
        workload: request.identity().map_err(invalid)?,
        topology_digest: Digest::from_bytes(*mesh.canonical_digest.bytes()),
        predecessor: request.predecessor.identity().map_err(invalid)?,
        start_interval: request.predecessor.next_interval(),
        end_interval: request.end_interval,
        charged_operator_work: prepared
            .reservation()
            .map_err(invalid)?
            .operator_work_upper_bound,
        phase_resident_bytes: ByteCount(resident),
        canonical_input_bytes: ByteCount(add(request_bytes.len() as u64, mesh_bytes.len() as u64)?),
        actual_decode_resident_bytes: ByteCount(decode),
        checkpoint_encoded_bytes: ByteCount(output.encoded_bytes_upper_bound),
        checkpoint_resident_bytes: ByteCount(output.materialization_resident_bytes_estimate),
    })
}
pub fn build_fluid_transient_block_resource_plan(
    request: &FluidTransientBlockWorkload,
    mesh: &SolverMeshArtifact,
    facts: PreparedPhaseResourceFacts,
    local_execution: LocalExecutionPlan,
) -> Result<FeaResourcePlan, ResourcePlanningError> {
    let projection = derive_fluid_transient_block_resource_projection(request, mesh)?;
    plan_from_projection(request, projection, facts, local_execution)
}
pub(crate) fn plan_from_projection(
    request: &FluidTransientBlockWorkload,
    projection: FluidTransientBlockResourceProjection,
    facts: PreparedPhaseResourceFacts,
    local_execution: LocalExecutionPlan,
) -> Result<FeaResourcePlan, ResourcePlanningError> {
    if facts.inputs.encoded_payload_bytes != projection.canonical_input_bytes
        || facts.inputs.decoded_resident_bytes_admission_budget
            < projection.actual_decode_resident_bytes
        || facts.output.encoded_payload_bytes_upper_bound != projection.checkpoint_encoded_bytes
        || facts.output.construction_resident_bytes_admission_budget
            < projection.checkpoint_resident_bytes
    {
        return Err(invalid("fluid transient source/output budget differs"));
    }
    let mut ceiling = request.workload.model.sequence.steps[0].accuracy.resources;
    ceiling.set_resident_byte_budget(request.workload.resident_ceiling().map_err(invalid)?);
    build_resource_plan(FeaResourcePlanningInput {
        estimation_model_revision: RESOURCE_ESTIMATION_MODEL_REVISION,
        phase: FeaResourcePhase::FluidTransientBlock(projection.workload),
        demand: FeaResourceDemand::FluidTransientBlock(FluidTransientBlockResourceDemand {
            projection,
            inputs: facts.inputs,
            output: facts.output,
        }),
        output_set: Some(facts.output_set),
        accuracy_ceiling: ceiling,
        worker: WorkerResourcePolicy {
            cpu_millicores: local_execution.checked_cpu_millicores().map_err(invalid)?,
            maximum_wall_milliseconds: ceiling.maximum_wall_time_milliseconds,
            fixed_memory_bytes: ByteCount(16 * 1024 * 1024),
            fixed_scratch_bytes: ByteCount(8 * 1024 * 1024),
            required_capabilities: Default::default(),
            accelerators: Vec::new(),
        },
    })
}

/// Sealed source-derived facts allow transport planning without replaying history twice.
pub struct PreparedFluidTransientResources {
    projection: FluidTransientBlockResourceProjection,
}
pub fn prepare_fluid_transient_resources(
    request: &FluidTransientBlockWorkload,
    mesh: &SolverMeshArtifact,
) -> Result<PreparedFluidTransientResources, ResourcePlanningError> {
    Ok(PreparedFluidTransientResources {
        projection: derive_fluid_transient_block_resource_projection(request, mesh)?,
    })
}
impl PreparedFluidTransientResources {
    pub fn projection(&self) -> &FluidTransientBlockResourceProjection {
        &self.projection
    }
    pub fn build(
        self,
        request: &FluidTransientBlockWorkload,
        facts: PreparedPhaseResourceFacts,
        local_execution: LocalExecutionPlan,
    ) -> Result<FeaResourcePlan, ResourcePlanningError> {
        if request.identity().map_err(invalid)? != self.projection.workload {
            return Err(invalid("fluid transient resource source changed"));
        }
        plan_from_projection(request, self.projection, facts, local_execution)
    }
}
