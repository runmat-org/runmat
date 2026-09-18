use crate::{
    algebra::CoercivitySymbolicPlan,
    constraints::CanonicalStructuralConstraintPlan,
    execution::{FieldRecoveryPlan, StructuralLinearWorkload},
    results::{field::StructuralRecoveredFieldTopology, observation::ResolvedObservationPlan},
    solve::preconditioner::SpdPreconditionerKind,
};
use runmat_execution::resource::AcceleratorRequest;
use runmat_meshing_core::SolverMeshArtifact;

use super::{
    build_resource_plan, structural_facts::derive_structural_resource_facts,
    AccuracyResourceDemand, AssemblyResourcePolicy, ByteCount, CheckpointArtifactPolicy, Count,
    FeaResourceDemand, FeaResourcePhase, FeaResourcePlan, FeaResourcePlanningInput,
    FieldRecoveryResourceFacts, NumericalScalar, ObservationResourceFacts, PartitioningPolicy,
    PhaseResourcePolicy, PreconditionerStorage, ResourcePlanningError, SolveResourcePolicy,
    SolverAdmissionResourcePolicy, StructuralOperatorResourceDemand, WorkerResourcePolicy,
    RESOURCE_ESTIMATION_MODEL_REVISION,
};

const STRUCTURAL_ASSEMBLY_SCRATCH_BYTES_PER_ELEMENT: u64 = 8 * 1024;
const STRUCTURAL_FIXED_ASSEMBLY_SCRATCH_BYTES: u64 = 64 * 1024;
const STRUCTURAL_FIXED_WORKER_MEMORY_BYTES: u64 = 16 * 1024 * 1024;
const STRUCTURAL_FIXED_WORKER_SCRATCH_BYTES: u64 = 8 * 1024 * 1024;
const STRUCTURAL_ARTIFACT_OVERHEAD_BYTES: u64 = 64 * 1024;
const STRUCTURAL_SOLVE_VECTOR_COUNT: u64 = 8;

/// Builds the canonical resource plan from authoritative post-projection numerical inputs.
///
/// This function does no assembly and accepts no caller-supplied topology, sparse, constraint, or
/// observation counts. It is therefore safe to run before allocations and expensive kernels.
pub fn build_structural_resource_plan(
    mesh: Option<&SolverMeshArtifact>,
    constraints: &CanonicalStructuralConstraintPlan,
    observations: &ResolvedObservationPlan,
    workload: &StructuralLinearWorkload,
    accelerator_requests: &[AcceleratorRequest],
) -> Result<FeaResourcePlan, ResourcePlanningError> {
    workload
        .validate()
        .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?;
    let facts = derive_structural_resource_facts(&workload.model, mesh, constraints)?;
    if observations.topology_digest != facts.topology_digest {
        return Err(ResourcePlanningError::Invalid(
            "observation plan and structural constraints name different topology".into(),
        ));
    }
    let contribution_nonzeros = u64::try_from(
        facts
            .symbolic
            .stiffness_upper_bound()
            .column_indices()
            .len(),
    )
    .map_err(|_| ResourcePlanningError::Overflow("assembly contribution count"))?;
    let contribution_bytes =
        contribution_nonzeros
            .checked_mul(24)
            .ok_or(ResourcePlanningError::Overflow(
                "assembly contribution bytes",
            ))?;
    let observations =
        ObservationResourceFacts::from_plan(observations, workload.observation_execution)?;
    let step = workload
        .procedure_step()
        .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?;
    let recovery_topology = StructuralRecoveredFieldTopology::derive(&workload.model, mesh)
        .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?;
    let field_recovery = FieldRecoveryResourceFacts::from_plan(
        &FieldRecoveryPlan::resolve(step, recovery_topology, constraints.groups.len())
            .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?,
    )?;
    let dofs = facts.sparse.rows.0;
    let stiffness_nonzeros = facts.sparse.maximum_nonzeros();
    let admission_pattern = if constraints.rows.iter().any(|row| row.terms.len() > 1) {
        facts
            .constraint_matrix
            .affine_reduction()
            .and_then(|reduction| {
                reduction.reduce_sparse_pattern(facts.symbolic.stiffness_upper_bound())
            })
            .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?
    } else {
        facts.symbolic.stiffness_upper_bound().clone()
    };
    let admission = CoercivitySymbolicPlan::derive(&admission_pattern)
        .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?;
    let solve = solve_policy(
        workload.preconditioner,
        dofs,
        stiffness_nonzeros,
        &admission,
    )?;
    let publishes_linearization = step
        .state_outputs
        .iter()
        .any(|output| output.kind == runmat_analysis_core::StateKind::GeometricStiffness);
    let recovery_values = facts
        .recovery_node_count
        .checked_mul(6)
        .and_then(|value| value.checked_add(facts.recovery_element_count.checked_mul(13)?))
        .ok_or(ResourcePlanningError::Overflow(
            "structural recovery value count",
        ))?;
    let mut recovery_values = u64::try_from(recovery_values)
        .map_err(|_| ResourcePlanningError::Overflow("structural recovery value count"))?;
    if publishes_linearization {
        // Two full vectors, worst-case four full stress tensors per element, and a complete
        // sparse geometric-stiffness CSR. Index storage is charged at the same eight-byte width.
        let linearization_values = dofs
            .checked_mul(2)
            .and_then(|value| value.checked_add(facts.topology.elements.0.checked_mul(36)?))
            .and_then(|value| value.checked_add(stiffness_nonzeros.checked_mul(2)?))
            .and_then(|value| value.checked_add(dofs.checked_add(1)?))
            .ok_or(ResourcePlanningError::Overflow(
                "structural linearization value count",
            ))?;
        recovery_values = recovery_values.checked_add(linearization_values).ok_or(
            ResourcePlanningError::Overflow("structural result value count"),
        )?;
    }
    let recovery_bytes = checked_bytes(recovery_values, 8, "structural recovery bytes")?;
    let recovery_resident = checked_add(
        recovery_bytes,
        STRUCTURAL_ARTIFACT_OVERHEAD_BYTES,
        "structural recovery resident bytes",
    )?;
    let checkpoint_bytes = checked_add(
        checked_bytes(dofs, 8, "structural checkpoint values")?,
        STRUCTURAL_ARTIFACT_OVERHEAD_BYTES,
        "structural checkpoint bytes",
    )?;
    let accelerators = structural_accelerator_requirements(workload, accelerator_requests)?;
    let required_capabilities = accelerators
        .iter()
        .map(|accelerator| {
            runmat_execution::resource::Capability::Accelerator(accelerator.class.clone())
        })
        .collect();
    build_resource_plan(FeaResourcePlanningInput {
        estimation_model_revision: RESOURCE_ESTIMATION_MODEL_REVISION,
        phase: FeaResourcePhase::StructuralLinearPipeline,
        demand: FeaResourceDemand::StructuralLinearPipeline(StructuralOperatorResourceDemand {
            topology_digest: facts.topology_digest,
            numerical_scope_digest: facts.symbolic.identity(),
            topology: facts.topology,
            sparse: facts.sparse,
            constraints: facts.constraints,
            assembly: AssemblyResourcePolicy {
                contribution_nonzero_upper_bound: Count(contribution_nonzeros),
                contribution_bytes_upper_bound: ByteCount(contribution_bytes),
                scratch_bytes_per_element: ByteCount(STRUCTURAL_ASSEMBLY_SCRATCH_BYTES_PER_ELEMENT),
                fixed_scratch_bytes_per_worker: ByteCount(STRUCTURAL_FIXED_ASSEMBLY_SCRATCH_BYTES),
            },
            solve,
            phase_resources: PhaseResourcePolicy {
                resident_value_count: Count(recovery_values),
                resident_bytes_upper_bound: ByteCount(recovery_resident),
                scratch_bytes: ByteCount(recovery_resident),
                retained_bytes: ByteCount(recovery_bytes),
            },
            field_recovery,
            observations: observations.clone(),
            publication: CheckpointArtifactPolicy {
                result_artifact_bytes: ByteCount(recovery_bytes),
                checkpoint_bytes: ByteCount(checkpoint_bytes),
                artifact_overhead_bytes: ByteCount(STRUCTURAL_ARTIFACT_OVERHEAD_BYTES),
                relay_bytes: ByteCount(0),
                result_egress_bytes: ByteCount(recovery_bytes),
                checkpoint_egress_bytes: ByteCount(checkpoint_bytes),
            },
            // The current structural kernel is monolithic. A recommendation would be dishonest
            // until the operator-domain execution path can consume it.
            partitioning: PartitioningPolicy::Monolithic,
            accuracy_demand: AccuracyResourceDemand {
                refinement_cycles: Count(0),
                solver_tightenings: Count(0),
                samples: observations.maximum_samples_per_observation,
            },
        }),
        output_set: None,
        worker: WorkerResourcePolicy {
            cpu_millicores: 1_000,
            maximum_wall_milliseconds: step.accuracy.resources.maximum_wall_time_milliseconds,
            fixed_memory_bytes: ByteCount(STRUCTURAL_FIXED_WORKER_MEMORY_BYTES),
            fixed_scratch_bytes: ByteCount(STRUCTURAL_FIXED_WORKER_SCRATCH_BYTES),
            required_capabilities,
            accelerators,
        },
        accuracy_ceiling: step.accuracy.resources,
    })
}

fn structural_accelerator_requirements(
    workload: &StructuralLinearWorkload,
    requests: &[AcceleratorRequest],
) -> Result<Vec<super::AcceleratorRequirement>, ResourcePlanningError> {
    if workload.backend != crate::ComputeBackend::Gpu {
        return Ok(Vec::new());
    }
    let [request] = requests else {
        return Err(ResourcePlanningError::Invalid(
            "GPU structural execution requires exactly one accelerator request".into(),
        ));
    };
    if request.count != 1 || request.memory_bytes_each == 0 {
        return Err(ResourcePlanningError::Invalid(
            "the single-device structural envelope requires one accelerator with bounded memory"
                .into(),
        ));
    }
    Ok(vec![super::AcceleratorRequirement {
        class: request.class.clone(),
        count: request.count,
        memory_bytes_each: ByteCount(request.memory_bytes_each),
    }])
}

fn solve_policy(
    preconditioner: SpdPreconditionerKind,
    dofs: u64,
    stiffness_nonzeros: u64,
    admission: &CoercivitySymbolicPlan,
) -> Result<SolveResourcePolicy, ResourcePlanningError> {
    let storage = match preconditioner {
        SpdPreconditionerKind::Jacobi => PreconditionerStorage::Diagonal,
        SpdPreconditionerKind::Ilu0 => {
            let row_offsets = checked_bytes(
                dofs.checked_add(1)
                    .ok_or(ResourcePlanningError::Overflow("ILU row offsets"))?,
                8,
                "ILU row offsets",
            )?;
            let entries = checked_bytes(stiffness_nonzeros, 16, "ILU columns and values")?;
            PreconditionerStorage::SparseCsr {
                nonzero_upper_bound: Count(stiffness_nonzeros),
                resident_bytes_upper_bound: ByteCount(checked_add(
                    row_offsets,
                    entries,
                    "ILU resident bytes",
                )?),
            }
        }
    };
    Ok(SolveResourcePolicy::Iterative {
        vector_count: Count(STRUCTURAL_SOLVE_VECTOR_COUNT),
        scalar: NumericalScalar::RealF64,
        preconditioner: storage,
        admission: SolverAdmissionResourcePolicy {
            dimension: Count(admission.dimension()),
            factor_nonzero_upper_bound: Count(admission.factor_nonzero_count()),
            update_count_upper_bound: Count(admission.update_count()),
            resident_bytes_upper_bound: ByteCount(admission.resident_bytes_upper_bound()),
        },
    })
}

fn checked_bytes(
    count: u64,
    width: u64,
    label: &'static str,
) -> Result<u64, ResourcePlanningError> {
    count
        .checked_mul(width)
        .ok_or(ResourcePlanningError::Overflow(label))
}

fn checked_add(left: u64, right: u64, label: &'static str) -> Result<u64, ResourcePlanningError> {
    left.checked_add(right)
        .ok_or(ResourcePlanningError::Overflow(label))
}
