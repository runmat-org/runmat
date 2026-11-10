use runmat_execution::resource::AcceleratorRequest;
use runmat_meshing_core::SolverMeshArtifact;

use crate::{
    algebra::CoercivityWorkEnvelope,
    constraints::CanonicalStructuralConstraintPlan,
    execution::{CanonicalSemanticState, StructuralBucklingWorkload},
    post::fields::structural_observable_field_definitions,
    results::observation::{resolve_structural_observation_plan, ObservationLimits},
    solve::buckling::BUCKLING_ORTHOGONALIZATION_PASSES,
};

use super::{
    assembly_sizing::{
        ASSEMBLY_SCRATCH_BYTES_PER_ELEMENT, FIXED_ASSEMBLY_SCRATCH_BYTES_PER_WORKER,
    },
    build_resource_plan,
    structural_facts::derive_structural_resource_facts,
    AccuracyResourceDemand, AssemblyResourcePolicy, ByteCount, CheckpointArtifactPolicy, Count,
    FeaResourceDemand, FeaResourcePhase, FeaResourcePlan, FeaResourcePlanningInput,
    ObservationResourceFacts, PartitioningPolicy, PhaseResourcePolicy, PreparedInputResourceFacts,
    PreparedOutputResourceFacts, PreparedOutputSetResourceFacts, ResourcePlanningError,
    SolveResourcePolicy, StructuralBucklingResourceDemand, StructuralOperatorResourceDemand,
    SubspaceSolvePath,
    WorkerResourcePolicy, RESOURCE_ESTIMATION_MODEL_REVISION,
};

const BOUNDARY_PROBE_WIDTH: u64 = 4;
const FIXED_WORKER_MEMORY_BYTES: u64 = 16 * 1024 * 1024;
const FIXED_WORKER_SCRATCH_BYTES: u64 = 8 * 1024 * 1024;
const RESULT_FIXED_BYTES: u64 = 64 * 1024;
const RECEIPT_BYTES: u64 = 16 * 1024;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreparedStructuralBucklingResourceFacts {
    pub local_execution: super::LocalExecutionPlan,
    pub inputs: PreparedInputResourceFacts,
    pub result_output: PreparedOutputResourceFacts,
    pub receipt_output: PreparedOutputResourceFacts,
    pub output_set: PreparedOutputSetResourceFacts,
}

pub fn build_structural_buckling_resource_plan(
    mesh: &SolverMeshArtifact,
    constraints: &CanonicalStructuralConstraintPlan,
    workload: &StructuralBucklingWorkload,
    prestress_state: &CanonicalSemanticState,
    accelerator_requests: &[AcceleratorRequest],
    prepared: PreparedStructuralBucklingResourceFacts,
) -> Result<FeaResourcePlan, ResourcePlanningError> {
    workload
        .validate()
        .map_err(|error| invalid(error.to_string()))?;
    prestress_state
        .validate()
        .map_err(|error| invalid(error.to_string()))?;
    if !accelerator_requests.is_empty() || workload.backend() != crate::ComputeBackend::Cpu {
        return Err(invalid(
            "structural buckling currently has no qualified accelerator implementation",
        ));
    }
    let linearization = prestress_state
        .payload
        .structural_linearization()
        .ok_or_else(|| invalid("buckling state has no structural linearization"))?;
    if prestress_state.identity != workload.prestress_transition().source_state
        || prestress_state.topology_digest != constraints.topology.discretization_digest()
        || linearization.topology_digest() != prestress_state.topology_digest
    {
        return Err(invalid(
            "buckling workload, preload state, and constraint topology differ",
        ));
    }
    let facts = derive_structural_resource_facts(workload.model(), Some(mesh), constraints)?;
    let fields =
        structural_observable_field_definitions().map_err(|error| invalid(error.to_string()))?;
    let observations = resolve_structural_observation_plan(
        workload
            .procedure_step()
            .map_err(|error| invalid(error.to_string()))?,
        workload.model(),
        Some(mesh),
        constraints,
        &fields,
        ObservationLimits::default(),
    )
    .map_err(|error| invalid(error.to_string()))?;
    let observations =
        ObservationResourceFacts::from_plan(&observations, workload.observation_execution())?;
    let reduction = facts
        .constraint_matrix
        .affine_reduction()
        .map_err(|error| invalid(error.to_string()))?;
    if reduction.offset.iter().any(|value| *value != 0.0) {
        return Err(invalid(
            "buckling perturbation constraints must have a homogeneous offset",
        ));
    }
    let full_dimension = count(reduction.full_dof_count, "buckling full dimension")?;
    let reduced_dimension = count(reduction.reduced_dof_count, "buckling reduced dimension")?;
    let requested = count(
        workload.solver().requested_mode_count(),
        "buckling requested modes",
    )?;
    if requested > reduced_dimension {
        return Err(invalid(
            "buckling requested mode count exceeds the reduced dimension",
        ));
    }
    let subspace = requested
        .checked_add(BOUNDARY_PROBE_WIDTH)
        .ok_or(ResourcePlanningError::Overflow("buckling subspace"))?
        .min(reduced_dimension);
    let reduced_stiffness = reduction
        .reduce_sparse_pattern(facts.symbolic.stiffness_upper_bound())
        .map_err(|error| invalid(error.to_string()))?;
    let reduced_geometric = reduction
        .reduce_sparse_pattern(linearization.geometric_stiffness().pattern())
        .map_err(|error| invalid(error.to_string()))?;
    let admission = CoercivityWorkEnvelope::derive(&reduced_stiffness)
        .map_err(|error| invalid(error.to_string()))?;
    let reduced_stiffness_nonzeros = count(
        reduced_stiffness.column_indices().len(),
        "reduced buckling stiffness nonzeros",
    )?;
    let reduced_geometric_nonzeros = count(
        reduced_geometric.column_indices().len(),
        "reduced geometric stiffness nonzeros",
    )?;
    // Shared block Lanczos on K⁻¹(−K_G): basis max(2s + 12, s + 48) with block 6, plus
    // restart/Ritz copies; projected matrices of the basis size.
    let basis = checked_add(
        checked_mul(subspace, 2, "buckling Lanczos basis")?,
        12,
        "buckling Lanczos basis",
    )?
    .max(checked_add(subspace, 48, "buckling Lanczos basis")?);
    let dense_projected = checked_mul(
        checked_mul(basis, basis, "buckling projected matrix")?,
        4,
        "buckling projected matrices",
    )?;
    let subspace_vectors = checked_mul(
        reduced_dimension,
        checked_add(
            checked_mul(basis, 2, "buckling basis copies")?,
            subspace + 36,
            "buckling vectors",
        )?,
        "buckling live Lanczos vectors",
    )?;
    let reduced_sparse_values = checked_add(
        reduced_stiffness_nonzeros,
        reduced_geometric_nonzeros,
        "buckling reduced sparse values",
    )?;
    let full_output_values = structural_buckling_output_value_count(full_dimension, requested)?;
    let phase_values = checked_add(
        checked_add(
            subspace_vectors,
            dense_projected,
            "buckling dense workspace",
        )?,
        checked_add(
            reduced_sparse_values,
            full_output_values,
            "buckling sparse and output values",
        )?,
        "buckling phase values",
    )?;
    let workspace_bytes = checked_add(
        checked_mul(phase_values, 8, "buckling phase bytes")?,
        admission.resident_bytes_limit(),
        "buckling coercivity workspace",
    )?;
    let result_payload = structural_buckling_result_payload_bytes_upper_bound(
        reduction.full_dof_count,
        workload.solver().requested_mode_count(),
    )?;
    let receipt_payload = structural_buckling_receipt_payload_bytes_upper_bound();
    if prepared.result_output.encoded_payload_bytes_upper_bound.0 != result_payload
        || prepared.receipt_output.encoded_payload_bytes_upper_bound.0 != receipt_payload
        || prepared.output_set.encoded_bytes_upper_bound.0 == 0
    {
        return Err(invalid(
            "prepared buckling outputs differ from their canonical bounds",
        ));
    }
    let output_transport = checked_add(
        prepared.result_output.transport_closure_bytes_upper_bound.0,
        prepared
            .receipt_output
            .transport_closure_bytes_upper_bound
            .0,
        "buckling output transport",
    )?;
    let output_payload = checked_add(result_payload, receipt_payload, "buckling output payload")?;
    let output_overhead = output_transport
        .checked_sub(output_payload)
        .ok_or_else(|| invalid("buckling output transport is below its payload"))?;
    let output_construction = checked_add(
        prepared
            .result_output
            .construction_resident_bytes_admission_budget
            .0,
        prepared
            .receipt_output
            .construction_resident_bytes_admission_budget
            .0,
        "buckling output construction",
    )?;
    let phase_resident = workspace_bytes.max(output_construction);
    let contribution_nonzeros = facts
        .sparse
        .operator_nonzeros
        .iter()
        .try_fold(0_u64, |total, value| {
            checked_add(total, value.0, "buckling assembly contributions")
        })?;
    let step = workload
        .procedure_step()
        .map_err(|error| invalid(error.to_string()))?;
    let scope_digest = runmat_execution::Digest::sha256(
        [
            b"runmat-structural-buckling-operator/v1\0".as_slice(),
            facts.symbolic.identity().bytes().as_slice(),
            linearization.identity().digest().bytes().as_slice(),
        ]
        .concat(),
    );

    build_resource_plan(FeaResourcePlanningInput {
        estimation_model_revision: RESOURCE_ESTIMATION_MODEL_REVISION,
        phase: FeaResourcePhase::StructuralBuckling(workload.identity()),
        demand: FeaResourceDemand::StructuralBuckling(StructuralBucklingResourceDemand {
            workload: workload.identity(),
            requested_mode_count: Count(requested),
            phase_value_count_upper_bound: Count(phase_values),
            output_value_count_upper_bound: Count(full_output_values),
            inputs: prepared.inputs.clone(),
            result_output: prepared.result_output,
            receipt_output: prepared.receipt_output,
            operator: StructuralOperatorResourceDemand {
                excitation_workspace: super::excitation_workspace::derive_excitation_workspace(
                    workload.model(),
                    workload
                        .procedure_step()
                        .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?,
                    Some(mesh),
                )?,
                topology_digest: facts.topology_digest,
                numerical_scope_digest: scope_digest,
                topology: facts.topology,
                sparse: facts.sparse,
                constraints: facts.constraints,
                assembly: AssemblyResourcePolicy {
                    contribution_nonzero_upper_bound: Count(contribution_nonzeros),
                    contribution_bytes_upper_bound: ByteCount(
                        super::connector_buffers::include_connector_buffers(
                            checked_mul(contribution_nonzeros, 24, "buckling contribution bytes")?,
                            facts.connector_assembly_buffer_bytes,
                        )?,
                    ),
                    scratch_bytes_per_element: ByteCount(ASSEMBLY_SCRATCH_BYTES_PER_ELEMENT),
                    fixed_scratch_bytes_per_worker: ByteCount(
                        FIXED_ASSEMBLY_SCRATCH_BYTES_PER_WORKER,
                    ),
                },
                solve: SolveResourcePolicy::SymmetricSubspaceIteration {
                    full_dimension: Count(full_dimension),
                    reduced_dimension: Count(reduced_dimension),
                    requested_mode_count: Count(requested),
                    initial_subspace_dimension: Count(subspace),
                    maximum_subspace_dimension: Count(reduced_dimension),
                    boundary_probe_width: Count(BOUNDARY_PROBE_WIDTH),
                    subspace_expansion_count_upper_bound: Count(1),
                    maximum_iterations_per_expansion: Count(count(
                        workload.solver().maximum_iterations(),
                        "buckling iteration limit",
                    )?),
                    orthogonalization_passes_per_block: Count(count(
                        BUCKLING_ORTHOGONALIZATION_PASSES,
                        "buckling orthogonalization passes",
                    )?),
                    dense_projected_value_count_upper_bound: Count(checked_mul(
                        reduced_dimension,
                        reduced_dimension,
                        "buckling maximum projected values",
                    )?),
                    workspace_bytes_upper_bound: ByteCount(workspace_bytes),
                    path: SubspaceSolvePath::ShiftInvertFactorization,
                },
                phase_resources: PhaseResourcePolicy {
                    resident_value_count: Count(phase_values),
                    resident_bytes_upper_bound: ByteCount(phase_resident),
                    scratch_bytes: ByteCount(phase_resident),
                    retained_bytes: ByteCount(output_payload),
                },
                field_recovery: Default::default(),
                observations: observations.clone(),
                publication: CheckpointArtifactPolicy {
                    result_artifact_bytes: ByteCount(output_payload),
                    checkpoint_bytes: ByteCount(0),
                    artifact_overhead_bytes: ByteCount(output_overhead),
                    relay_bytes: ByteCount(0),
                    result_egress_bytes: ByteCount(output_transport),
                    checkpoint_egress_bytes: ByteCount(0),
                },
                partitioning: PartitioningPolicy::Monolithic,
                accuracy_demand: AccuracyResourceDemand {
                    refinement_cycles: Count(0),
                    solver_tightenings: Count(0),
                    samples: observations.maximum_samples_per_observation,
                },
            },
        }),
        output_set: Some(prepared.output_set),
        worker: WorkerResourcePolicy {
            cpu_millicores: prepared
                .local_execution
                .checked_cpu_millicores()
                .map_err(|error| invalid(error.to_string()))?,
            maximum_wall_milliseconds: step.accuracy.resources.maximum_wall_time_milliseconds,
            fixed_memory_bytes: ByteCount(FIXED_WORKER_MEMORY_BYTES),
            fixed_scratch_bytes: ByteCount(FIXED_WORKER_SCRATCH_BYTES),
            required_capabilities: Default::default(),
            accelerators: Vec::new(),
        },
        accuracy_ceiling: step.accuracy.resources,
    })
}

pub fn structural_buckling_result_payload_bytes_upper_bound(
    full_dimension: usize,
    requested_mode_count: usize,
) -> Result<u64, ResourcePlanningError> {
    let values = structural_buckling_output_value_count(
        count(full_dimension, "buckling result dimension")?,
        count(requested_mode_count, "buckling result modes")?,
    )?;
    checked_add(
        checked_mul(values, 8, "buckling result values")?,
        RESULT_FIXED_BYTES,
        "buckling result payload",
    )
}

pub const fn structural_buckling_receipt_payload_bytes_upper_bound() -> u64 {
    RECEIPT_BYTES
}

fn structural_buckling_output_value_count(
    full_dimension: u64,
    requested_mode_count: u64,
) -> Result<u64, ResourcePlanningError> {
    checked_mul(
        requested_mode_count,
        full_dimension
            .checked_add(2)
            .ok_or(ResourcePlanningError::Overflow("buckling output width"))?,
        "buckling output values",
    )
}

fn count(value: usize, label: &'static str) -> Result<u64, ResourcePlanningError> {
    u64::try_from(value).map_err(|_| ResourcePlanningError::Overflow(label))
}

fn checked_mul(left: u64, right: u64, label: &'static str) -> Result<u64, ResourcePlanningError> {
    left.checked_mul(right)
        .ok_or(ResourcePlanningError::Overflow(label))
}

fn checked_add(left: u64, right: u64, label: &'static str) -> Result<u64, ResourcePlanningError> {
    left.checked_add(right)
        .ok_or(ResourcePlanningError::Overflow(label))
}

fn invalid(message: impl Into<String>) -> ResourcePlanningError {
    ResourcePlanningError::Invalid(message.into())
}
