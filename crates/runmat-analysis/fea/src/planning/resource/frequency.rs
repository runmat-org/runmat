use std::collections::BTreeSet;

use runmat_analysis_core::AccuracyResourceCeiling;
use runmat_execution::resource::AcceleratorRequest;
use runmat_execution::Digest;
use runmat_meshing_core::{CanonicalMeshingContract, SolverMeshArtifact};

use crate::{
    constraints::CanonicalStructuralConstraintPlan,
    execution::{
        FieldRecoveryRequest, StructuralFrequencyEntryRequest, StructuralFrequencyEntryResult,
        StructuralFrequencySolverPlan, StructuralFrequencyStateInputs, StructuralFrequencyWorkload,
    },
    post::{complex_structural_field_layout, ComplexStructuralFieldCardinality},
    ComputeBackend,
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
    PreparedOutputResourceFacts, PreparedOutputSetResourceFacts, PreparedPhaseResourceFacts,
    ResourcePlanningError, SolveResourcePolicy, StructuralFrequencyOperatorResourceDemand,
    StructuralOperatorResourceDemand, WorkerResourcePolicy, RESOURCE_ESTIMATION_MODEL_REVISION,
};

mod accounting;
mod payload;
mod recovery_plan;

use accounting::{direct_workspace_bytes, modal_workspace_bytes};
pub use payload::{
    structural_frequency_entry_payload_bytes_upper_bound,
    structural_frequency_recovery_payload_bytes_upper_bound,
};

#[derive(Clone, Copy)]
pub struct StructuralFrequencyRecoveryResourceInput<'a> {
    pub request: &'a FieldRecoveryRequest,
    pub result: &'a StructuralFrequencyEntryResult,
}

const FIXED_WORKER_MEMORY_BYTES: u64 = 16 * 1024 * 1024;
const FIXED_WORKER_SCRATCH_BYTES: u64 = 8 * 1024 * 1024;

fn structural_frequency_recovery_layout(
    mesh: &SolverMeshArtifact,
    constraints: &CanonicalStructuralConstraintPlan,
    workload: &StructuralFrequencyWorkload,
) -> Result<crate::post::ComplexStructuralFieldLayout, ResourcePlanningError> {
    let facts = derive_structural_resource_facts(workload.model(), Some(mesh), constraints)?;
    complex_structural_field_layout(ComplexStructuralFieldCardinality {
        structural_nodes: facts.recovery_node_count,
        solid_elements: facts.recovery_solid_element_count,
        beam_elements: facts.recovery_beam_element_count,
        shell_elements: facts.recovery_shell_element_count,
        connectors: workload.model().connectors.len(),
        constraint_groups: constraints.groups.len(),
        has_rotational_dofs: facts.recovery_has_rotational_dofs,
    })
    .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))
}

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
        states: StructuralFrequencyStateInputs<'_>,
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
        let topology_digest = runmat_execution::Digest::from_bytes(*mesh.canonical_digest.bytes());
        let linearization = match (workload.prestress_transition(), states.prestress) {
            (None, None) => None,
            (Some(transition), Some(state)) => Some(
                crate::pipeline::prestress::structural_linearization_from_state(
                    transition,
                    state,
                    topology_digest,
                )
                .map_err(ResourcePlanningError::Invalid)?,
            ),
            _ => {
                return Err(ResourcePlanningError::Invalid(
                    "frequency workload and prestress-state presence differ".into(),
                ))
            }
        };
        let facts = derive_structural_resource_facts(workload.model(), Some(mesh), constraints)?;
        let tangent_pattern = match linearization {
            Some(linearization) => facts
                .symbolic
                .stiffness_upper_bound()
                .union(linearization.geometric_stiffness().pattern())
                .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?,
            None => facts.symbolic.stiffness_upper_bound().clone(),
        };
        let sparse = super::SparseResourceFacts::from_patterns([
            &tangent_pattern,
            facts.symbolic.mass_upper_bound(),
            facts.symbolic.damping_upper_bound(),
        ])?;
        let numerical_scope_digest = match linearization {
            Some(linearization) => runmat_execution::Digest::sha256(
                [
                    b"runmat-structural-prestressed-frequency-operator/v1\0".as_slice(),
                    facts.symbolic.identity().bytes().as_slice(),
                    linearization.identity().digest().bytes().as_slice(),
                ]
                .concat(),
            ),
            None => facts.symbolic.identity(),
        };
        let full_dofs = facts.sparse.rows.0;
        let (solve, modal_operator_digest) = match (workload.solver(), states.modal_basis) {
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
                let reduced_patterns = [
                    &tangent_pattern,
                    facts.symbolic.mass_upper_bound(),
                    facts.symbolic.damping_upper_bound(),
                ]
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
            tangent_pattern.column_indices().len(),
            "frequency assembly contributions",
        )?;
        let contribution_bytes =
            multiply(contribution_nonzeros, 24, "assembly contribution bytes")?;
        let recovery_layout = structural_frequency_recovery_layout(mesh, constraints, workload)?;
        let step = workload
            .procedure_step()
            .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?;
        Ok(Self {
            workload: workload.clone(),
            topology_digest: facts.topology_digest,
            structural_topology: constraints.topology,
            numerical_scope_digest,
            topology: facts.topology,
            sparse,
            constraints: facts.constraints,
            assembly: AssemblyResourcePolicy {
                contribution_nonzero_upper_bound: Count(contribution_nonzeros),
                contribution_bytes_upper_bound: ByteCount(
                    super::connector_buffers::include_connector_buffers(
                        contribution_bytes,
                        facts.connector_quadrature_bytes,
                    )?,
                ),
                scratch_bytes_per_element: ByteCount(ASSEMBLY_SCRATCH_BYTES_PER_ELEMENT),
                fixed_scratch_bytes_per_worker: ByteCount(FIXED_ASSEMBLY_SCRATCH_BYTES_PER_WORKER),
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
        states: StructuralFrequencyStateInputs<'_>,
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
        let topology_digest = runmat_execution::Digest::from_bytes(*mesh.canonical_digest.bytes());
        if states.modal_basis.is_some() {
            return Err(ResourcePlanningError::Invalid(
                "frequency recovery does not consume a modal-basis state".into(),
            ));
        }
        let linearization = match (workload.prestress_transition(), states.prestress) {
            (None, None) => None,
            (Some(transition), Some(state)) => Some(
                crate::pipeline::prestress::structural_linearization_from_state(
                    transition,
                    state,
                    topology_digest,
                )
                .map_err(ResourcePlanningError::Invalid)?,
            ),
            _ => {
                return Err(ResourcePlanningError::Invalid(
                    "frequency recovery workload and prestress-state presence differ".into(),
                ))
            }
        };
        let facts = derive_structural_resource_facts(workload.model(), Some(mesh), constraints)?;
        let tangent_pattern = match linearization {
            Some(linearization) => facts
                .symbolic
                .stiffness_upper_bound()
                .union(linearization.geometric_stiffness().pattern())
                .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?,
            None => facts.symbolic.stiffness_upper_bound().clone(),
        };
        let sparse = super::SparseResourceFacts::from_patterns([
            &tangent_pattern,
            facts.symbolic.mass_upper_bound(),
            facts.symbolic.damping_upper_bound(),
        ])?;
        let numerical_scope_digest = match linearization {
            Some(linearization) => runmat_execution::Digest::sha256(
                [
                    b"runmat-structural-prestressed-frequency-recovery/v1\0".as_slice(),
                    facts.symbolic.identity().bytes().as_slice(),
                    linearization.identity().digest().bytes().as_slice(),
                ]
                .concat(),
            ),
            None => facts.symbolic.identity(),
        };
        let contribution_nonzeros = count(
            tangent_pattern.column_indices().len(),
            "frequency recovery assembly contributions",
        )?;
        let recovery_layout = structural_frequency_recovery_layout(mesh, constraints, workload)?;
        let step = workload
            .procedure_step()
            .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?;
        let full_dofs = facts.sparse.rows.0;
        Ok(Self {
            workload: workload.clone(),
            topology_digest: facts.topology_digest,
            structural_topology: constraints.topology,
            numerical_scope_digest,
            topology: facts.topology,
            sparse,
            constraints: facts.constraints,
            assembly: AssemblyResourcePolicy {
                contribution_nonzero_upper_bound: Count(contribution_nonzeros),
                contribution_bytes_upper_bound: ByteCount(
                    super::connector_buffers::include_connector_buffers(
                        multiply(
                            contribution_nonzeros,
                            24,
                            "frequency recovery contribution bytes",
                        )?,
                        facts.connector_quadrature_bytes,
                    )?,
                ),
                scratch_bytes_per_element: ByteCount(ASSEMBLY_SCRATCH_BYTES_PER_ELEMENT),
                fixed_scratch_bytes_per_worker: ByteCount(FIXED_ASSEMBLY_SCRATCH_BYTES_PER_WORKER),
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
                        field_recovery: Default::default(),
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
}

/// Builds the exact CPU resource envelope for one independently scheduled frequency entry.
pub fn build_structural_frequency_entry_resource_plan(
    mesh: &SolverMeshArtifact,
    constraints: &CanonicalStructuralConstraintPlan,
    workload: &StructuralFrequencyWorkload,
    request: &StructuralFrequencyEntryRequest,
    states: StructuralFrequencyStateInputs<'_>,
    accelerator_requests: &[AcceleratorRequest],
    prepared: PreparedPhaseResourceFacts,
) -> Result<FeaResourcePlan, ResourcePlanningError> {
    StructuralFrequencyEntryResourceProjection::new(
        mesh,
        constraints,
        workload,
        states,
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
    recovery: StructuralFrequencyRecoveryResourceInput<'_>,
    states: StructuralFrequencyStateInputs<'_>,
    accelerator_requests: &[AcceleratorRequest],
    prepared: PreparedPhaseResourceFacts,
) -> Result<FeaResourcePlan, ResourcePlanningError> {
    StructuralFrequencyEntryResourceProjection::for_recovery(
        mesh,
        constraints,
        workload,
        states,
        accelerator_requests,
    )?
    .recovery_plan_for(
        recovery.request,
        recovery.result,
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
