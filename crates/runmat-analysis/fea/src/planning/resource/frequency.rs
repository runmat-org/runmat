use std::collections::BTreeSet;

use runmat_analysis_core::AccuracyResourceCeiling;
use runmat_execution::resource::AcceleratorRequest;
use runmat_execution::Digest;
use runmat_meshing_core::{CanonicalMeshingContract, SolverMeshArtifact};

use crate::{
    constraints::CanonicalStructuralConstraintPlan,
    execution::{
        CanonicalSemanticState, FieldRecoveryRequest, FieldRecoverySource,
        StructuralFrequencyEntryRequest, StructuralFrequencyEntryResult,
        StructuralFrequencySolverPlan, StructuralFrequencyWorkload,
    },
    post::{complex_structural_field_layout, ComplexStructuralFieldCardinality},
    ComputeBackend,
};

use super::{
    build_resource_plan, structural_facts::derive_structural_resource_facts,
    AccuracyResourceDemand, AssemblyResourcePolicy, ByteCount, CheckpointArtifactPolicy, Count,
    FeaResourceDemand, FeaResourcePhase, FeaResourcePlan, FeaResourcePlanningInput,
    ObservationResourceFacts, PartitioningPolicy, PhaseResourcePolicy, PreparedInputResourceFacts,
    PreparedOutputResourceFacts, PreparedOutputSetResourceFacts, PreparedPhaseResourceFacts,
    ResourcePlanningError, SolveResourcePolicy, StructuralFrequencyOperatorResourceDemand,
    StructuralOperatorResourceDemand, WorkerResourcePolicy, RESOURCE_ESTIMATION_MODEL_REVISION,
};

mod accounting;
mod payload;

use accounting::{direct_workspace_bytes, modal_workspace_bytes};
use payload::recovery_payload_bytes_upper_bound;
pub use payload::{
    structural_frequency_entry_payload_bytes_upper_bound,
    structural_frequency_recovery_payload_bytes_upper_bound,
};

const ASSEMBLY_SCRATCH_BYTES_PER_ELEMENT: u64 = 8 * 1024;
const FIXED_ASSEMBLY_SCRATCH_BYTES: u64 = 64 * 1024;
const FIXED_WORKER_MEMORY_BYTES: u64 = 16 * 1024 * 1024;
const FIXED_WORKER_SCRATCH_BYTES: u64 = 8 * 1024 * 1024;

/// Assembly-free reusable projection for every independently scheduled entry in one frequency
/// workload. Expensive topology and reduced-pattern derivation occurs once per fan-out.
#[derive(Clone)]
pub struct StructuralFrequencyEntryResourceProjection {
    workload: StructuralFrequencyWorkload,
    topology_digest: Digest,
    structural_topology: crate::physics::structural::topology::CanonicalStructuralTopologyIdentity,
    numerical_scope_digest: Digest,
    topology: super::TopologyResourceFacts,
    sparse: super::SparseResourceFacts,
    constraints: super::ConstraintResourceFacts,
    assembly: AssemblyResourcePolicy,
    solve: SolveResourcePolicy,
    modal_operator_digest: Option<Digest>,
    recovery_layout: crate::post::ComplexStructuralFieldLayout,
    constraint_group_count: usize,
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
        let recovery_layout = complex_structural_field_layout(ComplexStructuralFieldCardinality {
            structural_nodes: facts.recovery_node_count,
            solid_elements: facts.recovery_solid_element_count,
            beam_elements: facts.recovery_beam_element_count,
            shell_elements: facts.recovery_shell_element_count,
            constraint_groups: constraints.groups.len(),
            has_rotational_dofs: facts.recovery_has_rotational_dofs,
        })
        .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?;
        let step = workload
            .procedure_step()
            .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?;
        Ok(Self {
            workload: workload.clone(),
            topology_digest: facts.topology_digest,
            structural_topology: constraints.topology,
            numerical_scope_digest: facts.symbolic.identity(),
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
            recovery_layout,
            constraint_group_count: constraints.groups.len(),
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

    pub(super) fn for_recovery(
        mesh: &SolverMeshArtifact,
        constraints: &CanonicalStructuralConstraintPlan,
        workload: &StructuralFrequencyWorkload,
        accelerator_requests: &[AcceleratorRequest],
    ) -> Result<Self, ResourcePlanningError> {
        workload
            .validate()
            .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?;
        mesh.validate_canonical()
            .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?;
        if workload.backend() != ComputeBackend::Cpu || !accelerator_requests.is_empty() {
            return Err(ResourcePlanningError::Invalid(
                "structural frequency recovery currently requires CPU resources without accelerators"
                    .into(),
            ));
        }
        let facts = derive_structural_resource_facts(workload.model(), Some(mesh), constraints)?;
        let contribution_nonzeros = count(
            facts
                .symbolic
                .stiffness_upper_bound()
                .column_indices()
                .len(),
            "frequency recovery assembly contributions",
        )?;
        let recovery_layout = complex_structural_field_layout(ComplexStructuralFieldCardinality {
            structural_nodes: facts.recovery_node_count,
            solid_elements: facts.recovery_solid_element_count,
            beam_elements: facts.recovery_beam_element_count,
            shell_elements: facts.recovery_shell_element_count,
            constraint_groups: constraints.groups.len(),
            has_rotational_dofs: facts.recovery_has_rotational_dofs,
        })
        .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?;
        let step = workload
            .procedure_step()
            .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?;
        let full_dofs = facts.sparse.rows.0;
        Ok(Self {
            workload: workload.clone(),
            topology_digest: facts.topology_digest,
            structural_topology: constraints.topology,
            numerical_scope_digest: facts.symbolic.identity(),
            topology: facts.topology,
            sparse: facts.sparse,
            constraints: facts.constraints,
            assembly: AssemblyResourcePolicy {
                contribution_nonzero_upper_bound: Count(contribution_nonzeros),
                contribution_bytes_upper_bound: ByteCount(multiply(
                    contribution_nonzeros,
                    24,
                    "frequency recovery contribution bytes",
                )?),
                scratch_bytes_per_element: ByteCount(ASSEMBLY_SCRATCH_BYTES_PER_ELEMENT),
                fixed_scratch_bytes_per_worker: ByteCount(FIXED_ASSEMBLY_SCRATCH_BYTES),
            },
            solve: SolveResourcePolicy::NoSolve {
                workspace_bytes_upper_bound: ByteCount(multiply(
                    full_dofs,
                    32,
                    "complex recovery operator workspace",
                )?),
            },
            modal_operator_digest: None,
            recovery_layout,
            constraint_group_count: constraints.groups.len(),
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
        inputs: PreparedInputResourceFacts,
        output: PreparedOutputResourceFacts,
        output_set: PreparedOutputSetResourceFacts,
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
        let result_bytes = structural_frequency_entry_payload_bytes_upper_bound(request)?;
        if output.encoded_payload_bytes_upper_bound.0 != result_bytes
            || output.transport_closure_bytes_upper_bound.0 < result_bytes
            || output.construction_resident_bytes_admission_budget.0 < result_bytes
        {
            return Err(ResourcePlanningError::Invalid(
                "frequency entry output facts differ from its canonical payload ceiling".into(),
            ));
        }
        let artifact_overhead_bytes = output
            .transport_closure_bytes_upper_bound
            .0
            .checked_sub(result_bytes)
            .ok_or_else(|| {
                ResourcePlanningError::Invalid(
                    "frequency entry transport closure is smaller than its payload".into(),
                )
            })?;
        build_resource_plan(FeaResourcePlanningInput {
            estimation_model_revision: RESOURCE_ESTIMATION_MODEL_REVISION,
            phase: FeaResourcePhase::StructuralFrequencyEntry(request.identity()),
            demand: FeaResourceDemand::StructuralFrequencyEntry(
                StructuralFrequencyOperatorResourceDemand {
                    inputs,
                    output: output.clone(),
                    operator: StructuralOperatorResourceDemand {
                        topology_digest: self.topology_digest,
                        numerical_scope_digest: self.numerical_scope_digest,
                        topology: self.topology.clone(),
                        sparse: self.sparse.clone(),
                        constraints: self.constraints.clone(),
                        assembly: self.assembly.clone(),
                        solve: self.solve.clone(),
                        phase_resources: PhaseResourcePolicy {
                            resident_value_count: Count(0),
                            resident_bytes_upper_bound: output
                                .construction_resident_bytes_admission_budget,
                            scratch_bytes: ByteCount(0),
                            retained_bytes: ByteCount(result_bytes),
                        },
                        observations: empty_observations(),
                        publication: CheckpointArtifactPolicy {
                            result_artifact_bytes: ByteCount(result_bytes),
                            checkpoint_bytes: ByteCount(0),
                            artifact_overhead_bytes: ByteCount(artifact_overhead_bytes),
                            relay_bytes: ByteCount(0),
                            result_egress_bytes: output.transport_closure_bytes_upper_bound,
                            checkpoint_egress_bytes: ByteCount(0),
                        },
                        partitioning: PartitioningPolicy::Monolithic,
                        accuracy_demand: AccuracyResourceDemand {
                            refinement_cycles: Count(0),
                            solver_tightenings: Count(0),
                            samples: Count(1),
                        },
                    },
                },
            ),
            output_set: Some(output_set),
            worker: self.worker.clone(),
            accuracy_ceiling: self.accuracy_ceiling,
        })
    }

    pub fn recovery_plan_for(
        &self,
        request: &FieldRecoveryRequest,
        result: &StructuralFrequencyEntryResult,
        inputs: PreparedInputResourceFacts,
        output: PreparedOutputResourceFacts,
        output_set: PreparedOutputSetResourceFacts,
    ) -> Result<FeaResourcePlan, ResourcePlanningError> {
        request
            .validate()
            .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?;
        result
            .validate()
            .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?;
        result
            .request()
            .validate_for_workload(&self.workload)
            .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?;
        if request.producer != result.request().step()
            || request.source != FieldRecoverySource::StructuralFrequencyEntry(result.identity())
            || request.topology != self.structural_topology
        {
            return Err(ResourcePlanningError::Invalid(
                "frequency recovery request differs from its workload, entry, or topology".into(),
            ));
        }
        let full_dofs = self.sparse.rows.0;
        let output_values = count(
            self.recovery_layout.storage_value_count,
            "complex recovery output values",
        )?;
        let constraint_rows = count(
            self.constraint_group_count,
            "complex recovery constraint groups",
        )?;
        let recovery_values = add(
            multiply(
                output_values,
                2,
                "real, imaginary, and complex recovery values",
            )?,
            add(
                multiply(
                    full_dofs,
                    4,
                    "complex displacement and reaction work values",
                )?,
                multiply(constraint_rows, 12, "complex constraint recovery values")?,
                "complex recovery work values",
            )?,
            "complex recovery live values",
        )?;
        let recovery_bytes = multiply(recovery_values, 8, "complex recovery live bytes")?;
        let output_payload_bytes = recovery_payload_bytes_upper_bound(&self.recovery_layout)?;
        if output.encoded_payload_bytes_upper_bound.0 != output_payload_bytes
            || output.transport_closure_bytes_upper_bound.0 < output_payload_bytes
            || output.construction_resident_bytes_admission_budget.0 < output_payload_bytes
        {
            return Err(ResourcePlanningError::Invalid(
                "frequency recovery output facts differ from its canonical payload ceiling".into(),
            ));
        }
        let output_overhead = output
            .transport_closure_bytes_upper_bound
            .0
            .checked_sub(output_payload_bytes)
            .ok_or_else(|| {
                ResourcePlanningError::Invalid(
                    "frequency recovery closure is smaller than its payload".into(),
                )
            })?;
        let resident_bytes = add(
            recovery_bytes,
            output.construction_resident_bytes_admission_budget.0,
            "complex recovery output construction residency",
        )?;
        build_resource_plan(FeaResourcePlanningInput {
            estimation_model_revision: RESOURCE_ESTIMATION_MODEL_REVISION,
            phase: FeaResourcePhase::StructuralFrequencyFieldRecovery(request.identity()),
            demand: FeaResourceDemand::StructuralFrequencyFieldRecovery(
                StructuralFrequencyOperatorResourceDemand {
                    inputs,
                    output: output.clone(),
                    operator: StructuralOperatorResourceDemand {
                        topology_digest: self.topology_digest,
                        numerical_scope_digest: self.numerical_scope_digest,
                        topology: self.topology.clone(),
                        sparse: self.sparse.clone(),
                        constraints: self.constraints.clone(),
                        assembly: self.assembly.clone(),
                        solve: SolveResourcePolicy::NoSolve {
                            workspace_bytes_upper_bound: ByteCount(multiply(
                                full_dofs,
                                32,
                                "complex recovery operator workspace",
                            )?),
                        },
                        phase_resources: PhaseResourcePolicy {
                            resident_value_count: Count(recovery_values),
                            resident_bytes_upper_bound: ByteCount(resident_bytes),
                            scratch_bytes: ByteCount(0),
                            retained_bytes: ByteCount(multiply(
                                output_values,
                                8,
                                "complex recovery retained bytes",
                            )?),
                        },
                        observations: empty_observations(),
                        publication: CheckpointArtifactPolicy {
                            result_artifact_bytes: ByteCount(output_payload_bytes),
                            checkpoint_bytes: ByteCount(0),
                            artifact_overhead_bytes: ByteCount(output_overhead),
                            relay_bytes: ByteCount(0),
                            result_egress_bytes: output.transport_closure_bytes_upper_bound,
                            checkpoint_egress_bytes: ByteCount(0),
                        },
                        partitioning: PartitioningPolicy::Monolithic,
                        accuracy_demand: AccuracyResourceDemand {
                            refinement_cycles: Count(0),
                            solver_tightenings: Count(0),
                            samples: Count(1),
                        },
                    },
                },
            ),
            output_set: Some(output_set),
            worker: self.worker.clone(),
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
    prepared: PreparedPhaseResourceFacts,
) -> Result<FeaResourcePlan, ResourcePlanningError> {
    StructuralFrequencyEntryResourceProjection::new(
        mesh,
        constraints,
        workload,
        modal_state,
        accelerator_requests,
    )?
    .plan_for(
        request,
        prepared.inputs,
        prepared.output,
        prepared.output_set,
    )
}

/// Builds the exact CPU resource envelope for one independently scheduled frequency recovery.
pub fn build_structural_frequency_recovery_resource_plan(
    mesh: &SolverMeshArtifact,
    constraints: &CanonicalStructuralConstraintPlan,
    workload: &StructuralFrequencyWorkload,
    request: &FieldRecoveryRequest,
    result: &StructuralFrequencyEntryResult,
    accelerator_requests: &[AcceleratorRequest],
    prepared: PreparedPhaseResourceFacts,
) -> Result<FeaResourcePlan, ResourcePlanningError> {
    StructuralFrequencyEntryResourceProjection::for_recovery(
        mesh,
        constraints,
        workload,
        accelerator_requests,
    )?
    .recovery_plan_for(
        request,
        result,
        prepared.inputs,
        prepared.output,
        prepared.output_set,
    )
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
