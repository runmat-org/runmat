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
    LocalExecutionPlan, ObservationResourceFacts, PartitioningPolicy, PhaseResourcePolicy,
    PreparedInputResourceFacts, PreparedOutputResourceFacts, PreparedOutputSetResourceFacts,
    PreparedPhaseResourceFacts, ResourcePlanningError, SolveResourcePolicy,
    StructuralFrequencyOperatorResourceDemand, StructuralOperatorResourceDemand,
    WorkerResourcePolicy, RESOURCE_ESTIMATION_MODEL_REVISION,
};

mod accounting;
mod admission;
mod projection;
pub use admission::{
    build_structural_frequency_entry_resource_plan,
    build_structural_frequency_entry_resource_plan_from_compilation,
    build_structural_frequency_recovery_resource_plan,
    build_structural_frequency_recovery_resource_plan_from_compilation,
};
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
        nodes: facts.recovery_node_count,
        volume_elements: facts.recovery_solid_element_count,
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
    constraint_compilation_peak_bytes: ByteCount,
    workload: StructuralFrequencyWorkload,
    topology_digest: Digest,
    structural_topology: crate::physics::structural::topology::CanonicalStructuralTopologyIdentity,
    numerical_scope_digest: Digest,
    topology: super::TopologyResourceFacts,
    sparse: super::SparseResourceFacts,
    constraints: super::ConstraintResourceFacts,
    assembly: AssemblyResourcePolicy,
    excitation_workspace: super::ExcitationWorkspaceResourceFacts,
    solve: SolveResourcePolicy,
    modal_operator_digest: Option<Digest>,
    recovery_layout: crate::post::ComplexStructuralFieldLayout,
    constraint_group_count: usize,
    maximum_wall_milliseconds: u64,
    accuracy_ceiling: AccuracyResourceCeiling,
}

impl StructuralFrequencyEntryResourceProjection {
    /// One worker sized by the admitted local execution plan; the projection itself is
    /// thread-independent so one fan-out projection serves every entry.
    fn worker(
        &self,
        local_execution: LocalExecutionPlan,
    ) -> Result<WorkerResourcePolicy, ResourcePlanningError> {
        Ok(WorkerResourcePolicy {
            cpu_millicores: local_execution
                .checked_cpu_millicores()
                .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?,
            maximum_wall_milliseconds: self.maximum_wall_milliseconds,
            fixed_memory_bytes: ByteCount(FIXED_WORKER_MEMORY_BYTES),
            fixed_scratch_bytes: ByteCount(FIXED_WORKER_SCRATCH_BYTES),
            required_capabilities: BTreeSet::new(),
            accelerators: Vec::new(),
        })
    }

    pub fn plan_for(
        &self,
        request: &StructuralFrequencyEntryRequest,
        inputs: PreparedInputResourceFacts,
        output: PreparedOutputResourceFacts,
        output_set: PreparedOutputSetResourceFacts,
        local_execution: LocalExecutionPlan,
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
                    constraint_compilation_peak_bytes: self.constraint_compilation_peak_bytes,
                    inputs,
                    output: output.clone(),
                    operator: StructuralOperatorResourceDemand {
                        excitation_workspace: self.excitation_workspace.clone(),
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
            worker: self.worker(local_execution)?,
            accuracy_ceiling: self.accuracy_ceiling,
        })
    }
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
