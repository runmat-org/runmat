use std::collections::BTreeSet;

use runmat_analysis_core::AccuracyResourceCeiling;
use runmat_execution::resource::AcceleratorRequest;
use runmat_execution::Digest;
use runmat_meshing_core::{CanonicalMeshingContract, SolverMeshArtifact};

use crate::{
    constraints::CanonicalStructuralConstraintPlan,
    execution::{
        CanonicalSemanticState, StructuralFrequencyEntryRequest, StructuralFrequencySolverPlan,
        StructuralFrequencyWorkload,
    },
    ComputeBackend,
};

use super::{
    build_resource_plan, structural_facts::derive_structural_resource_facts,
    AccuracyResourceDemand, AssemblyResourcePolicy, ByteCount, CheckpointArtifactPolicy, Count,
    FeaResourcePhase, FeaResourcePlan, FeaResourcePlanningInput, ObservationResourceFacts,
    PartitioningPolicy, RecoveryResourcePolicy, ResourcePlanningError, SolveResourcePolicy,
    WorkerResourcePolicy, RESOURCE_ESTIMATION_MODEL_REVISION,
};

const ASSEMBLY_SCRATCH_BYTES_PER_ELEMENT: u64 = 8 * 1024;
const FIXED_ASSEMBLY_SCRATCH_BYTES: u64 = 64 * 1024;
const FIXED_WORKER_MEMORY_BYTES: u64 = 16 * 1024 * 1024;
const FIXED_WORKER_SCRATCH_BYTES: u64 = 8 * 1024 * 1024;
const ARTIFACT_OVERHEAD_BYTES: u64 = 64 * 1024;

/// Assembly-free reusable projection for every independently scheduled entry in one frequency
/// workload. Expensive topology and reduced-pattern derivation occurs once per fan-out.
#[derive(Clone)]
pub struct StructuralFrequencyEntryResourceProjection {
    workload: StructuralFrequencyWorkload,
    topology_digest: Digest,
    operator_pattern_digest: Digest,
    topology: super::TopologyResourceFacts,
    sparse: super::SparseResourceFacts,
    constraints: super::ConstraintResourceFacts,
    assembly: AssemblyResourcePolicy,
    solve: SolveResourcePolicy,
    modal_operator_digest: Option<Digest>,
    worker: WorkerResourcePolicy,
    accuracy_ceiling: AccuracyResourceCeiling,
}

impl StructuralFrequencyEntryResourceProjection {
    pub fn new(
        mesh: &SolverMeshArtifact,
        constraints: &CanonicalStructuralConstraintPlan,
        workload: &StructuralFrequencyWorkload,
        modal_state: Option<&CanonicalSemanticState>,
        accelerator_requests: &[AcceleratorRequest],
    ) -> Result<Self, ResourcePlanningError> {
        workload
            .validate()
            .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?;
        mesh.validate_canonical()
            .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?;
        if workload.backend() != ComputeBackend::Cpu || !accelerator_requests.is_empty() {
            return Err(ResourcePlanningError::Invalid(
                "structural frequency execution currently requires CPU resources without accelerators"
                    .into(),
            ));
        }
        let facts = derive_structural_resource_facts(workload.model(), Some(mesh), constraints)?;
        let full_dofs = facts.sparse.rows.0;
        let (solve, modal_operator_digest) = match (workload.solver(), modal_state) {
            (StructuralFrequencySolverPlan::Direct { policy }, None) => {
                let reduction = facts
                    .constraint_matrix
                    .affine_reduction()
                    .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?;
                if reduction.offset.iter().any(|value| *value != 0.0) {
                    return Err(ResourcePlanningError::Invalid(
                        "direct harmonic resource admission requires homogeneous constraints"
                            .into(),
                    ));
                }
                let reduced_dofs = count(reduction.reduced_dof_count, "reduced frequency DOFs")?;
                let block_dimension = multiply(reduced_dofs, 2, "real block dimension")?;
                let restart = count(policy.restart, "GMRES restart")?
                    .min(block_dimension)
                    .min(count(
                        policy.maximum_iterations,
                        "GMRES maximum iterations",
                    )?);
                let reduced_patterns = facts
                    .symbolic
                    .patterns()
                    .into_iter()
                    .map(|pattern| reduction.reduce_sparse_pattern(pattern))
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?;
                let reduced_nonzeros =
                    reduced_patterns.iter().try_fold(0_u64, |total, pattern| {
                        add(
                            total,
                            count(pattern.column_indices().len(), "reduced operator nonzeros")?,
                            "reduced operator nonzeros",
                        )
                    })?;
                (
                    SolveResourcePolicy::RestartedGmres {
                        block_dimension: Count(block_dimension),
                        restart: Count(restart),
                        maximum_iterations: Count(count(
                            policy.maximum_iterations,
                            "GMRES maximum iterations",
                        )?),
                        workspace_bytes_upper_bound: ByteCount(direct_workspace_bytes(
                            full_dofs,
                            reduced_dofs,
                            reduced_nonzeros,
                            &reduction,
                            restart,
                        )?),
                    },
                    None,
                )
            }
            (StructuralFrequencySolverPlan::ModalReduced { basis_state }, Some(state)) => {
                state
                    .validate()
                    .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?;
                let topology_digest =
                    runmat_execution::Digest::from_bytes(*mesh.canonical_digest.bytes());
                let basis = state.payload.modal_basis().ok_or_else(|| {
                    ResourcePlanningError::Invalid(
                        "modal harmonic resource admission requires a modal basis state".into(),
                    )
                })?;
                if state.identity != *basis_state
                    || state.topology_digest != topology_digest
                    || basis
                        .modes()
                        .iter()
                        .any(|mode| u64::try_from(mode.dof_shape().len()).ok() != Some(full_dofs))
                {
                    return Err(ResourcePlanningError::Invalid(
                        "modal harmonic basis identity or topology differs from the frequency task"
                            .into(),
                    ));
                }
                let mode_count = count(basis.modes().len(), "modal harmonic mode count")?;
                (
                    SolveResourcePolicy::ModalDenseHarmonic {
                        full_dimension: Count(full_dofs),
                        mode_count: Count(mode_count),
                        workspace_bytes_upper_bound: ByteCount(modal_workspace_bytes(
                            full_dofs, mode_count,
                        )?),
                    },
                    Some(basis.operator_digest()),
                )
            }
            (StructuralFrequencySolverPlan::Direct { .. }, Some(_)) => {
                return Err(ResourcePlanningError::Invalid(
                    "direct frequency resource admission received a modal state".into(),
                ));
            }
            (StructuralFrequencySolverPlan::ModalReduced { .. }, None) => {
                return Err(ResourcePlanningError::Invalid(
                    "modal frequency resource admission is missing its basis state".into(),
                ));
            }
        };
        let contribution_nonzeros = count(
            facts
                .symbolic
                .stiffness_upper_bound()
                .column_indices()
                .len(),
            "frequency assembly contributions",
        )?;
        let contribution_bytes =
            multiply(contribution_nonzeros, 24, "assembly contribution bytes")?;
        let step = workload
            .procedure_step()
            .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?;
        Ok(Self {
            workload: workload.clone(),
            topology_digest: facts.topology_digest,
            operator_pattern_digest: facts.symbolic.identity(),
            topology: facts.topology,
            sparse: facts.sparse,
            constraints: facts.constraints,
            assembly: AssemblyResourcePolicy {
                contribution_nonzero_upper_bound: Count(contribution_nonzeros),
                contribution_bytes_upper_bound: ByteCount(contribution_bytes),
                scratch_bytes_per_element: ByteCount(ASSEMBLY_SCRATCH_BYTES_PER_ELEMENT),
                fixed_scratch_bytes_per_worker: ByteCount(FIXED_ASSEMBLY_SCRATCH_BYTES),
            },
            solve,
            modal_operator_digest,
            worker: WorkerResourcePolicy {
                cpu_millicores: 1_000,
                maximum_wall_milliseconds: step.accuracy.resources.maximum_wall_time_milliseconds,
                fixed_memory_bytes: ByteCount(FIXED_WORKER_MEMORY_BYTES),
                fixed_scratch_bytes: ByteCount(FIXED_WORKER_SCRATCH_BYTES),
                required_capabilities: BTreeSet::new(),
                accelerators: Vec::new(),
            },
            accuracy_ceiling: step.accuracy.resources,
        })
    }

    pub fn plan_for(
        &self,
        request: &StructuralFrequencyEntryRequest,
    ) -> Result<FeaResourcePlan, ResourcePlanningError> {
        request
            .validate_for_workload(&self.workload)
            .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?;
        let full_dofs = self.sparse.rows.0;
        if count(
            request.scope().physical_dof_count(),
            "frequency request physical DOFs",
        )? != full_dofs
            || self
                .modal_operator_digest
                .is_some_and(|digest| digest != request.scope().operator_digest())
        {
            return Err(ResourcePlanningError::Invalid(
                "frequency request, symbolic topology, and modal basis are incompatible".into(),
            ));
        }
        let result_bytes = multiply(full_dofs, 32, "complex frequency result vectors")?;
        let result_egress_bytes = add(
            result_bytes,
            ARTIFACT_OVERHEAD_BYTES,
            "frequency result egress bytes",
        )?;
        build_resource_plan(FeaResourcePlanningInput {
            estimation_model_revision: RESOURCE_ESTIMATION_MODEL_REVISION,
            phase: FeaResourcePhase::StructuralFrequencyEntry(request.identity()),
            topology_digest: self.topology_digest,
            operator_pattern_digest: self.operator_pattern_digest,
            topology: self.topology.clone(),
            sparse: self.sparse.clone(),
            constraints: self.constraints.clone(),
            assembly: self.assembly.clone(),
            solve: self.solve.clone(),
            recovery: RecoveryResourcePolicy {
                resident_value_count: Count(0),
                resident_bytes_upper_bound: ByteCount(0),
                scratch_bytes: ByteCount(0),
                retained_bytes: ByteCount(0),
            },
            observations: empty_observations(),
            publication: CheckpointArtifactPolicy {
                result_artifact_bytes: ByteCount(result_bytes),
                checkpoint_bytes: ByteCount(0),
                artifact_overhead_bytes: ByteCount(ARTIFACT_OVERHEAD_BYTES),
                relay_bytes: ByteCount(0),
                result_egress_bytes: ByteCount(result_egress_bytes),
                checkpoint_egress_bytes: ByteCount(0),
            },
            worker: self.worker.clone(),
            partitioning: PartitioningPolicy::Monolithic,
            accuracy_demand: AccuracyResourceDemand {
                refinement_cycles: Count(0),
                solver_tightenings: Count(0),
                samples: Count(1),
            },
            accuracy_ceiling: self.accuracy_ceiling,
        })
    }
}

/// Builds the exact CPU resource envelope for one independently scheduled frequency entry.
pub fn build_structural_frequency_entry_resource_plan(
    mesh: &SolverMeshArtifact,
    constraints: &CanonicalStructuralConstraintPlan,
    workload: &StructuralFrequencyWorkload,
    request: &StructuralFrequencyEntryRequest,
    modal_state: Option<&CanonicalSemanticState>,
    accelerator_requests: &[AcceleratorRequest],
) -> Result<FeaResourcePlan, ResourcePlanningError> {
    StructuralFrequencyEntryResourceProjection::new(
        mesh,
        constraints,
        workload,
        modal_state,
        accelerator_requests,
    )?
    .plan_for(request)
}

fn direct_workspace_bytes(
    full_dofs: u64,
    reduced_dofs: u64,
    reduced_nonzeros: u64,
    reduction: &crate::constraints::AffineConstraintReduction,
    restart: u64,
) -> Result<u64, ResourcePlanningError> {
    let block_dimension = multiply(reduced_dofs, 2, "real block dimension")?;
    let reduced_csr = add(
        multiply(
            add(
                multiply(reduced_dofs, 3, "reduced CSR rows")?,
                3,
                "reduced CSR rows",
            )?,
            8,
            "reduced CSR row bytes",
        )?,
        multiply(reduced_nonzeros, 16, "reduced CSR entry bytes")?,
        "reduced CSR bytes",
    )?;
    let reduction_indices = [
        reduction.free_dof_indices.len(),
        reduction.pivot_dof_indices.len(),
        reduction.recovery_row_offsets.len(),
        reduction.recovery_column_indices.len(),
    ]
    .into_iter()
    .try_fold(0_u64, |total, value| {
        add(
            total,
            count(value, "affine reduction indices")?,
            "affine reduction indices",
        )
    })?;
    let reduction_values = add(
        count(
            reduction.recovery_coefficients.len(),
            "affine reduction coefficients",
        )?,
        count(reduction.offset.len(), "affine reduction offsets")?,
        "affine reduction values",
    )?;
    let reduction_bytes = multiply(
        add(
            reduction_indices,
            reduction_values,
            "affine reduction storage",
        )?,
        8,
        "affine reduction bytes",
    )?;
    let krylov_vectors = multiply(
        add(restart, 9, "GMRES live vector count")?,
        block_dimension,
        "GMRES live vector values",
    )?;
    let hessenberg = multiply(
        add(restart, 1, "Hessenberg rows")?,
        restart,
        "Hessenberg values",
    )?;
    let small_vectors = add(
        multiply(restart, 3, "GMRES rotations and coefficients")?,
        1,
        "GMRES projected RHS",
    )?;
    let full_order_vectors = multiply(full_dofs, 14, "full-order harmonic work vectors")?;
    let floating_values = add(
        add(krylov_vectors, hessenberg, "direct harmonic workspace")?,
        add(
            small_vectors,
            full_order_vectors,
            "direct harmonic workspace",
        )?,
        "direct harmonic workspace",
    )?;
    add(
        add(reduced_csr, reduction_bytes, "direct harmonic workspace")?,
        multiply(floating_values, 8, "direct harmonic floating workspace")?,
        "direct harmonic workspace",
    )
}

fn modal_workspace_bytes(full_dofs: u64, mode_count: u64) -> Result<u64, ResourcePlanningError> {
    let full_by_modes = multiply(full_dofs, mode_count, "modal shape values")?;
    let dense = multiply(mode_count, mode_count, "modal dense values")?;
    let values = add(
        multiply(full_by_modes, 4, "modal projection workspaces")?,
        add(
            multiply(dense, 11, "modal dense workspaces")?,
            add(
                multiply(full_dofs, 14, "modal full-order work vectors")?,
                multiply(mode_count, 8, "modal reduced work vectors")?,
                "modal vector workspaces",
            )?,
            "modal dense and vector workspaces",
        )?,
        "modal harmonic workspace",
    )?;
    multiply(values, 8, "modal harmonic workspace bytes")
}

fn empty_observations() -> ObservationResourceFacts {
    ObservationResourceFacts {
        maximum_samples_per_observation: Count(0),
        produced_sample_count: Count(0),
        retained_sample_count: Count(0),
        operation_count: Count(0),
        peak_resident_bytes: ByteCount(0),
        stored_bytes: ByteCount(0),
        transfer_bytes: ByteCount(0),
        collective_count: Count(0),
    }
}

fn count(value: usize, label: &'static str) -> Result<u64, ResourcePlanningError> {
    u64::try_from(value).map_err(|_| ResourcePlanningError::Overflow(label))
}

fn multiply(left: u64, right: u64, label: &'static str) -> Result<u64, ResourcePlanningError> {
    left.checked_mul(right)
        .ok_or(ResourcePlanningError::Overflow(label))
}

fn add(left: u64, right: u64, label: &'static str) -> Result<u64, ResourcePlanningError> {
    left.checked_add(right)
        .ok_or(ResourcePlanningError::Overflow(label))
}
