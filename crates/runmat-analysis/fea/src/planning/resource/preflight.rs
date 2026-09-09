use crate::{
    constraints::CanonicalStructuralConstraintPlan, execution::StructuralLinearWorkload,
    results::observation::ResolvedObservationPlan, solve::preconditioner::SpdPreconditionerKind,
    topology::AnalysisProjection,
};

use super::{
    build_resource_plan, AccuracyResourceDemand, AssemblyResourcePolicy, ByteCount,
    CheckpointArtifactPolicy, ConstraintResourceFacts, Count, FeaResourcePlan,
    FeaResourcePlanningInput, NumericalScalar, ObservationResourceFacts, PartitioningPolicy,
    PreconditionerStorage, RecoveryResourcePolicy, ResourcePlanningError, SolveResourcePolicy,
    SparseResourceFacts, StructuralSymbolicOperators, TopologyResourceFacts, WorkerResourcePolicy,
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
    projection: &AnalysisProjection,
    constraints: &CanonicalStructuralConstraintPlan,
    observations: &ResolvedObservationPlan,
    workload: &StructuralLinearWorkload,
) -> Result<FeaResourcePlan, ResourcePlanningError> {
    workload
        .validate()
        .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?;
    if observations.topology_digest != projection.projection_digest {
        return Err(ResourcePlanningError::Invalid(
            "observation plan and analysis projection name different topology".into(),
        ));
    }
    let symbolic = StructuralSymbolicOperators::derive(projection, constraints)?;
    let sparse = SparseResourceFacts::from_patterns(symbolic.patterns())?;
    let contribution_nonzeros =
        u64::try_from(symbolic.stiffness_upper_bound().column_indices().len())
            .map_err(|_| ResourcePlanningError::Overflow("assembly contribution count"))?;
    let contribution_bytes =
        contribution_nonzeros
            .checked_mul(24)
            .ok_or(ResourcePlanningError::Overflow(
                "assembly contribution bytes",
            ))?;
    let observations = ObservationResourceFacts::from_plan(observations)?;
    let dofs = sparse.rows.0;
    let stiffness_nonzeros = sparse.maximum_nonzeros();
    let solve = solve_policy(workload.preconditioner, dofs, stiffness_nonzeros)?;
    let recovery_values = projection
        .nodes
        .len()
        .checked_mul(6)
        .and_then(|value| value.checked_add(projection.elements.len().checked_mul(13)?))
        .ok_or(ResourcePlanningError::Overflow(
            "structural recovery value count",
        ))?;
    let recovery_values = u64::try_from(recovery_values)
        .map_err(|_| ResourcePlanningError::Overflow("structural recovery value count"))?;
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
    let step = workload
        .procedure_step()
        .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?;
    build_resource_plan(FeaResourcePlanningInput {
        estimation_model_revision: RESOURCE_ESTIMATION_MODEL_REVISION,
        topology_digest: projection.projection_digest,
        operator_pattern_digest: symbolic.identity(),
        topology: TopologyResourceFacts::from_projection(projection)?,
        sparse,
        constraints: ConstraintResourceFacts::from_plan(constraints)?,
        assembly: AssemblyResourcePolicy {
            contribution_nonzero_upper_bound: Count(contribution_nonzeros),
            contribution_bytes_upper_bound: ByteCount(contribution_bytes),
            scratch_bytes_per_element: ByteCount(STRUCTURAL_ASSEMBLY_SCRATCH_BYTES_PER_ELEMENT),
            fixed_scratch_bytes_per_worker: ByteCount(STRUCTURAL_FIXED_ASSEMBLY_SCRATCH_BYTES),
        },
        solve,
        recovery: RecoveryResourcePolicy {
            resident_value_count: Count(recovery_values),
            resident_bytes_upper_bound: ByteCount(recovery_resident),
            scratch_bytes: ByteCount(recovery_resident),
            retained_bytes: ByteCount(recovery_bytes),
        },
        observations: observations.clone(),
        publication: CheckpointArtifactPolicy {
            checkpoint_bytes: ByteCount(checkpoint_bytes),
            artifact_overhead_bytes: ByteCount(STRUCTURAL_ARTIFACT_OVERHEAD_BYTES),
            relay_bytes: ByteCount(0),
            result_egress_bytes: ByteCount(recovery_bytes),
            checkpoint_egress_bytes: ByteCount(checkpoint_bytes),
        },
        worker: WorkerResourcePolicy {
            cpu_millicores: 1_000,
            maximum_wall_milliseconds: step.accuracy.resources.maximum_wall_time_milliseconds,
            fixed_memory_bytes: ByteCount(STRUCTURAL_FIXED_WORKER_MEMORY_BYTES),
            fixed_scratch_bytes: ByteCount(STRUCTURAL_FIXED_WORKER_SCRATCH_BYTES),
            required_capabilities: Default::default(),
            accelerators: Vec::new(),
        },
        // The current structural kernel is monolithic. A recommendation would be dishonest until
        // the operator-domain execution path can consume it.
        partitioning: PartitioningPolicy::Monolithic,
        accuracy_demand: AccuracyResourceDemand {
            refinement_cycles: Count(0),
            solver_tightenings: Count(0),
            samples: observations.maximum_samples_per_observation,
        },
        accuracy_ceiling: step.accuracy.resources,
    })
}

fn solve_policy(
    preconditioner: SpdPreconditionerKind,
    dofs: u64,
    stiffness_nonzeros: u64,
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
