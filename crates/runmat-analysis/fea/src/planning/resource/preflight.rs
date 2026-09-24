use crate::{
    algebra::CoercivitySymbolicPlan,
    constraints::{
        AffineConstraintReduction, CanonicalStructuralConstraintPlan, ConstraintEnforcement,
    },
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
    FieldRecoveryResourceFacts, LocalExecutionPlan, NumericalScalar, ObservationResourceFacts,
    PartitioningPolicy, PhaseResourcePolicy, PreconditionerStorage, ResourcePlanningError,
    SolveResourcePolicy, SolverAdmissionResourcePolicy, StructuralOperatorResourceDemand,
    WorkerResourcePolicy, RESOURCE_ESTIMATION_MODEL_REVISION,
};

// Scoped solid or explicit beam/shell assembly retains at most eight contributions per worker.
// Tet10's two 30x30 f64 matrices (~14.1 KiB) dominate the beam payload; a shell retains three
// 18x18 f64 matrices at most (~7.6 KiB), DOF indices, and recovery. The 8 KiB per element plus
// 128 KiB per worker bounds the retained wave and element-local construction scratch. Sparse
// rows are counted separately.
const STRUCTURAL_ASSEMBLY_SCRATCH_BYTES_PER_ELEMENT: u64 = 8 * 1024;
const STRUCTURAL_FIXED_ASSEMBLY_SCRATCH_BYTES: u64 = 128 * 1024;
const STRUCTURAL_FIXED_WORKER_MEMORY_BYTES: u64 = 16 * 1024 * 1024;
const STRUCTURAL_FIXED_WORKER_SCRATCH_BYTES: u64 = 8 * 1024 * 1024;
const STRUCTURAL_ARTIFACT_OVERHEAD_BYTES: u64 = 64 * 1024;
// Scoped SpMV retains one complete set of chunk outputs while merging its result vector.
const STRUCTURAL_SOLVE_VECTOR_COUNT: u64 = 9;
// Five recovered solid arrays retain 61 f64 values per element (6 strain, 6 stress,
// 1 energy density, and 4 x 6 each of integration-point strain and stress). Scoped
// recovery can temporarily own one complete set of chunk arrays while merging them.
// Explicit beam and shell resultants retain nine f64 values per element and at
// most one additional nine-value chunk set during scoped merging, below this bound.
// One further scalar covers the element von-Mises field. Nodal displacement, reaction,
// and projected von-Mises require seven values per node.
const STRUCTURAL_RECOVERY_PEAK_VALUES_PER_ELEMENT: usize = 123;
const STRUCTURAL_RECOVERY_VALUES_PER_NODE: usize = 7;

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
    local_execution: LocalExecutionPlan,
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
    let contribution_bytes = super::connector_buffers::include_connector_buffers(
        contribution_nonzeros
            .checked_mul(24)
            .ok_or(ResourcePlanningError::Overflow(
                "assembly contribution bytes",
            ))?,
        facts.connector_quadrature_bytes,
    )?;
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
        u64::try_from(facts.constraint_matrix.row_count())
            .map_err(|_| ResourcePlanningError::Overflow("constraint row count"))?,
        constraints.enforcement,
        stiffness_nonzeros,
        &admission,
        if constraints.enforcement == ConstraintEnforcement::AffineReduction
            && !local_execution.is_serial()
        {
            let reduction = facts
                .constraint_matrix
                .affine_reduction()
                .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?;
            affine_transpose_index_peak_bytes(&reduction)?
        } else {
            0
        },
    )?;
    let publishes_linearization = step
        .state_outputs
        .iter()
        .any(|output| output.kind == runmat_analysis_core::StateKind::GeometricStiffness);
    let recovery_values = facts
        .recovery_node_count
        .checked_mul(STRUCTURAL_RECOVERY_VALUES_PER_NODE)
        .and_then(|value| {
            value.checked_add(
                facts
                    .recovery_element_count
                    .checked_mul(STRUCTURAL_RECOVERY_PEAK_VALUES_PER_ELEMENT)?,
            )
        })
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
    let cpu_millicores = local_execution
        .checked_cpu_millicores()
        .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?;
    let fixed_assembly_scratch = local_execution
        .checked_fixed_scratch_bytes(ByteCount(STRUCTURAL_FIXED_ASSEMBLY_SCRATCH_BYTES))
        .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?;
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
                fixed_scratch_bytes_per_worker: fixed_assembly_scratch,
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
            cpu_millicores,
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
    constraint_rows: u64,
    enforcement: ConstraintEnforcement,
    stiffness_nonzeros: u64,
    admission: &CoercivitySymbolicPlan,
    transpose_index_bytes: u64,
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
        vector_count: Count(structural_solve_vector_count(
            dofs,
            constraint_rows,
            enforcement,
        )?),
        scalar: NumericalScalar::RealF64,
        preconditioner: storage,
        admission: SolverAdmissionResourcePolicy {
            dimension: Count(admission.dimension()),
            factor_nonzero_upper_bound: Count(admission.factor_nonzero_count()),
            update_count_upper_bound: Count(admission.update_count()),
            resident_bytes_upper_bound: ByteCount(admission.resident_bytes_upper_bound()),
        },
        transpose_index_bytes_upper_bound: ByteCount(transpose_index_bytes),
    })
}

/// Bounds the retained restarted-GMRES basis and saddle work in units of full physical vectors.
/// The sixteen auxiliary saddle vectors cover RHS/initial/solution/residual, an in-flight
/// Arnoldi vector, operator output, constraint transforms and temporary reductions. Two full
/// physical vectors cover scoped CSR chunk output plus its ordered merge. Hessenberg, Givens,
/// and projected-RHS scalars are charged separately before rounding up to physical vectors.
fn structural_solve_vector_count(
    dofs: u64,
    constraint_rows: u64,
    enforcement: ConstraintEnforcement,
) -> Result<u64, ResourcePlanningError> {
    if enforcement != ConstraintEnforcement::LagrangeMultiplier {
        return Ok(STRUCTURAL_SOLVE_VECTOR_COUNT);
    }
    if dofs == 0 {
        return Err(ResourcePlanningError::Invalid(
            "multiplier solve requires physical DOFs".into(),
        ));
    }
    let saddle = dofs
        .checked_add(constraint_rows)
        .ok_or(ResourcePlanningError::Overflow("saddle dimension"))?;
    let restart = saddle.min(32);
    let basis_and_auxiliary =
        saddle
            .checked_mul(restart + 17)
            .ok_or(ResourcePlanningError::Overflow(
                "GMRES basis and work vectors",
            ))?;
    let scalar_work = (restart + 1)
        .checked_mul(restart + 6)
        .ok_or(ResourcePlanningError::Overflow("GMRES Hessenberg work"))?;
    let spmv_work = dofs
        .checked_mul(2)
        .ok_or(ResourcePlanningError::Overflow("scoped saddle SpMV work"))?;
    let values = basis_and_auxiliary
        .checked_add(scalar_work)
        .and_then(|sum| sum.checked_add(spmv_work))
        .ok_or(ResourcePlanningError::Overflow("multiplier solve values"))?;
    Ok(values.div_ceil(dofs).max(STRUCTURAL_SOLVE_VECTOR_COUNT))
}

fn affine_transpose_index_peak_bytes(
    reduction: &AffineConstraintReduction,
) -> Result<u64, ResourcePlanningError> {
    let entries = u64::try_from(reduction.recovery_column_indices.len())
        .map_err(|_| ResourcePlanningError::Overflow("affine transpose entries"))?;
    let columns = u64::try_from(reduction.reduced_dof_count)
        .map_err(|_| ResourcePlanningError::Overflow("affine transpose columns"))?;
    checked_add(
        checked_bytes(entries, 16, "affine transpose row-entry pairs")?,
        checked_bytes(
            columns
                .checked_add(1)
                .ok_or(ResourcePlanningError::Overflow("affine transpose offsets"))?,
            16,
            "affine transpose offsets and construction cursors",
        )?,
        "affine transpose index peak bytes",
    )
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn multiplier_gmres_basis_is_charged_above_scoped_pcg() {
        assert_eq!(
            structural_solve_vector_count(12, 7, ConstraintEnforcement::LagrangeMultiplier)
                .unwrap(),
            101
        );
        assert_eq!(
            structural_solve_vector_count(12, 7, ConstraintEnforcement::AffineReduction).unwrap(),
            STRUCTURAL_SOLVE_VECTOR_COUNT
        );
        assert!(structural_solve_vector_count(
            u64::MAX,
            1,
            ConstraintEnforcement::LagrangeMultiplier
        )
        .is_err());
    }

    #[test]
    fn affine_transpose_index_charges_entries_offsets_and_build_cursors() {
        let reduction = AffineConstraintReduction {
            full_dof_count: 4,
            reduced_dof_count: 2,
            free_dof_indices: vec![0, 1],
            pivot_dof_indices: vec![2, 3],
            recovery_row_offsets: vec![0, 1, 2, 4, 5],
            recovery_column_indices: vec![0, 1, 0, 1, 1],
            recovery_coefficients: vec![1.0; 5],
            offset: vec![0.0; 4],
        };
        reduction.validate().unwrap();
        assert_eq!(
            affine_transpose_index_peak_bytes(&reduction).unwrap(),
            5 * 16 + 3 * 16
        );
    }
}
