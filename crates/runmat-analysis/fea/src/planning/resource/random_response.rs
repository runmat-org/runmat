use std::collections::BTreeSet;

use runmat_meshing_core::{CanonicalMeshingContract, SolverMeshArtifact};

use crate::{
    constraints::CanonicalStructuralConstraintPlan,
    execution::{
        CanonicalSemanticState, StructuralRandomResponseAssemblyRequest,
        StructuralRandomResponseIntervalRequest, StructuralRandomResponseWorkload,
    },
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
    ResourcePlanningError, SolveResourcePolicy, StructuralOperatorResourceDemand,
    StructuralRandomResponseAssemblyResourceDemand, StructuralRandomResponseIntervalResourceDemand,
    WorkerResourcePolicy, RESOURCE_ESTIMATION_MODEL_REVISION,
};

const FIXED_WORKER_MEMORY_BYTES: u64 = 16 * 1024 * 1024;
const FIXED_WORKER_SCRATCH_BYTES: u64 = 8 * 1024 * 1024;
const INTERVAL_FIXED_PAYLOAD_BYTES: u64 = 256 * 1024;
const INTERVAL_METADATA_BYTES: u64 = 64 * 1024;
const ASSEMBLY_FIXED_PAYLOAD_BYTES: u64 = 256 * 1024;
const RECEIPT_PAYLOAD_BYTES: u64 = 16 * 1024;

/// Reusable exact projection for every independently scheduled interval task in one random step.
#[derive(Clone)]
pub struct StructuralRandomResponseResourceProjection {
    workload: StructuralRandomResponseWorkload,
    topology_digest: runmat_execution::Digest,
    topology: super::TopologyResourceFacts,
    sparse: super::SparseResourceFacts,
    constraints: super::ConstraintResourceFacts,
    assembly: AssemblyResourcePolicy,
    excitation_workspace: super::ExcitationWorkspaceResourceFacts,
    basis: crate::ModalBasisIdentity,
    operator_digest: runmat_execution::Digest,
    physical_dof_count: u64,
    available_mode_count: u64,
    maximum_wall_milliseconds: u64,
    accuracy_ceiling: runmat_analysis_core::AccuracyResourceCeiling,
}

impl StructuralRandomResponseResourceProjection {
    /// One worker sized by the admitted local execution plan; the projection itself is
    /// thread-independent so one projection serves every scheduled phase.
    fn worker(
        &self,
        local_execution: LocalExecutionPlan,
    ) -> Result<WorkerResourcePolicy, ResourcePlanningError> {
        Ok(WorkerResourcePolicy {
            cpu_millicores: local_execution
                .checked_cpu_millicores()
                .map_err(invalid_error)?,
            maximum_wall_milliseconds: self.maximum_wall_milliseconds,
            fixed_memory_bytes: ByteCount(FIXED_WORKER_MEMORY_BYTES),
            fixed_scratch_bytes: ByteCount(FIXED_WORKER_SCRATCH_BYTES),
            required_capabilities: BTreeSet::new(),
            accelerators: Vec::new(),
        })
    }

    pub fn new(
        mesh: &SolverMeshArtifact,
        constraints: &CanonicalStructuralConstraintPlan,
        workload: &StructuralRandomResponseWorkload,
        modal_state: &CanonicalSemanticState,
    ) -> Result<Self, ResourcePlanningError> {
        workload.validate().map_err(invalid_error)?;
        modal_state.validate().map_err(invalid_error)?;
        mesh.validate_canonical().map_err(invalid_error)?;
        if workload.backend() != ComputeBackend::Cpu {
            return Err(invalid(
                "random-response execution currently requires qualified CPU resources",
            ));
        }
        let topology_digest = runmat_execution::Digest::from_bytes(*mesh.canonical_digest.bytes());
        let basis = modal_state.payload.modal_basis().ok_or_else(|| {
            invalid("random-response resource admission requires a modal-basis state")
        })?;
        if modal_state.identity != workload.modal_basis_transition().source_state
            || modal_state.topology_digest != topology_digest
            || workload.modal_basis_transition().target_topology_digest != topology_digest
        {
            return Err(invalid(
                "random-response workload, state, and mesh topology differ",
            ));
        }
        let facts = derive_structural_resource_facts(workload.model(), Some(mesh), constraints)?;
        let physical_dof_count = facts.sparse.rows.0;
        if basis
            .modes()
            .iter()
            .any(|mode| u64::try_from(mode.dof_shape().len()).ok() != Some(physical_dof_count))
        {
            return Err(invalid(
                "random-response modal basis and structural DOF counts differ",
            ));
        }
        let available_mode_count = count(basis.modes().len(), "random-response mode count")?;
        let sparse = super::SparseResourceFacts::from_patterns([
            facts.symbolic.stiffness_upper_bound(),
            facts.symbolic.mass_upper_bound(),
            facts.symbolic.damping_upper_bound(),
        ])?;
        let contribution_nonzeros = facts
            .sparse
            .operator_nonzeros
            .iter()
            .try_fold(0_u64, |sum, value| {
                add(sum, value.0, "random-response operator contributions")
            })?;
        let step = workload.procedure_step().map_err(invalid_error)?;
        Ok(Self {
            excitation_workspace: super::excitation_workspace::derive_excitation_workspace(
                workload.model(),
                step,
                Some(mesh),
            )?,
            workload: workload.clone(),
            topology_digest,
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
                            "random-response operator contribution bytes",
                        )?,
                        facts.connector_assembly_buffer_bytes,
                    )?,
                ),
                scratch_bytes_per_element: ByteCount(ASSEMBLY_SCRATCH_BYTES_PER_ELEMENT),
                fixed_scratch_bytes_per_worker: ByteCount(FIXED_ASSEMBLY_SCRATCH_BYTES_PER_WORKER),
            },
            basis: basis.identity(),
            operator_digest: basis.operator_digest(),
            physical_dof_count,
            available_mode_count,
            maximum_wall_milliseconds: step.accuracy.resources.maximum_wall_time_milliseconds,
            accuracy_ceiling: step.accuracy.resources,
        })
    }

    pub fn plan_for_intervals(
        &self,
        request: &StructuralRandomResponseIntervalRequest,
        inputs: PreparedInputResourceFacts,
        output: PreparedOutputResourceFacts,
        output_set: PreparedOutputSetResourceFacts,
        local_execution: LocalExecutionPlan,
    ) -> Result<FeaResourcePlan, ResourcePlanningError> {
        request
            .validate_for(&self.workload)
            .map_err(invalid_error)?;
        let scope = request.scope();
        let plan = scope.plan();
        if scope.topology_digest() != self.topology_digest
            || scope.basis() != self.basis
            || scope.operator_digest() != self.operator_digest
            || count(plan.physical_dof_count(), "random-response physical DOFs")?
                != self.physical_dof_count
            || count(plan.selected_mode_count(), "random-response selected modes")?
                > self.available_mode_count
        {
            return Err(invalid(
                "random-response interval request differs from its admitted topology or basis",
            ));
        }
        let interval_count = request.interval_range()[1] - request.interval_range()[0];
        let interval_count = count(interval_count, "random-response task intervals")?;
        let channel_count = count(
            self.workload
                .excitation()
                .map_err(invalid_error)?
                .channels
                .len(),
            "random-response channels",
        )?;
        let spectrum = self.workload.excitation().map_err(invalid_error)?;
        let spectral_density = spectrum
            .resolved_spectral_density()
            .map_err(invalid_error)?;
        let spectral_matrix_complex_value_count = multiply(
            count(spectral_density.len(), "random-response spectral samples")?,
            multiply(
                channel_count,
                channel_count,
                "random-response spectral matrix width",
            )?,
            "random-response spectral matrix values",
        )?;
        let selected_mode_count =
            count(plan.selected_mode_count(), "random-response selected modes")?;
        let response_value_count = count(plan.value_count(), "random-response values")?;
        let linear_field_value_count = response_value_count
            .checked_sub(self.physical_dof_count)
            .ok_or(ResourcePlanningError::Overflow(
                "random-response linear field values",
            ))?;
        let evaluations_per_interval = (1_u64
            .checked_shl(u32::from(plan.policy().maximum_refinement_levels))
            .ok_or(ResourcePlanningError::Overflow(
                "random-response adaptive panels",
            ))?)
        .checked_add(1)
        .ok_or(ResourcePlanningError::Overflow(
            "random-response adaptive evaluations",
        ))?;
        let maximum_frequency_evaluation_count = multiply(
            interval_count,
            evaluations_per_interval,
            "random-response task frequency evaluations",
        )?;

        // Modal field recovery, complex channel transfers, interpolated PSD, adaptive cache, and
        // quadrature accumulator vectors coexist in the current physical interval kernel.
        let recovered_modal_values = multiply(
            selected_mode_count,
            response_value_count,
            "random-response recovered modal values",
        )?;
        let complex_transfer_values = multiply(
            2,
            add(
                add(
                    multiply(
                        selected_mode_count,
                        channel_count,
                        "random-response modal channel transfers",
                    )?,
                    multiply(
                        response_value_count,
                        channel_count,
                        "random-response physical channel transfers",
                    )?,
                    "random-response transfer values",
                )?,
                multiply(
                    channel_count,
                    channel_count,
                    "random-response interpolated PSD values",
                )?,
                "random-response complex workspace values",
            )?,
            "random-response complex scalar storage",
        )?;
        let quadrature_values = add(
            multiply(
                evaluations_per_interval,
                response_value_count,
                "random-response cached integrand values",
            )?,
            multiply(
                response_value_count,
                4,
                "random-response quadrature vectors",
            )?,
            "random-response quadrature workspace values",
        )?;
        let workspace_values = add(
            add(
                recovered_modal_values,
                complex_transfer_values,
                "random-response recovery and transfer values",
            )?,
            quadrature_values,
            "random-response total workspace values",
        )?;
        let workspace_bytes = multiply(workspace_values, 8, "random-response workspace bytes")?;
        let result_bytes = structural_random_response_interval_payload_bytes_upper_bound(request)?;
        validate_output(&output, result_bytes, "random-response interval result")?;
        let artifact_overhead = output
            .transport_closure_bytes_upper_bound
            .0
            .checked_sub(result_bytes)
            .ok_or_else(|| invalid("random-response interval transport is below its payload"))?;
        build_resource_plan(FeaResourcePlanningInput {
            estimation_model_revision: RESOURCE_ESTIMATION_MODEL_REVISION,
            phase: FeaResourcePhase::StructuralRandomResponseInterval(request.identity()),
            demand: FeaResourceDemand::StructuralRandomResponseInterval(
                StructuralRandomResponseIntervalResourceDemand {
                    operator: StructuralOperatorResourceDemand {
                        excitation_workspace: self.excitation_workspace.clone(),
                        topology_digest: self.topology_digest,
                        numerical_scope_digest: scope.operator_digest(),
                        topology: self.topology.clone(),
                        sparse: self.sparse.clone(),
                        constraints: self.constraints.clone(),
                        assembly: self.assembly.clone(),
                        solve: SolveResourcePolicy::NoSolve {
                            workspace_bytes_upper_bound: ByteCount(workspace_bytes),
                        },
                        phase_resources: PhaseResourcePolicy {
                            resident_value_count: Count(multiply(
                                interval_count,
                                response_value_count,
                                "random-response retained interval values",
                            )?),
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
                            artifact_overhead_bytes: ByteCount(artifact_overhead),
                            relay_bytes: ByteCount(0),
                            result_egress_bytes: output.transport_closure_bytes_upper_bound,
                            checkpoint_egress_bytes: ByteCount(0),
                        },
                        partitioning: PartitioningPolicy::Monolithic,
                        accuracy_demand: AccuracyResourceDemand {
                            refinement_cycles: Count(0),
                            solver_tightenings: Count(0),
                            samples: Count(maximum_frequency_evaluation_count),
                        },
                    },
                    interval_count: Count(interval_count),
                    channel_count: Count(channel_count),
                    spectral_matrix_complex_value_count: Count(spectral_matrix_complex_value_count),
                    selected_mode_count: Count(selected_mode_count),
                    physical_dof_count: Count(self.physical_dof_count),
                    linear_field_value_count: Count(linear_field_value_count),
                    response_value_count: Count(response_value_count),
                    maximum_frequency_evaluation_count: Count(maximum_frequency_evaluation_count),
                    inputs,
                    output,
                },
            ),
            output_set: Some(output_set),
            worker: self.worker(local_execution)?,
            accuracy_ceiling: self.accuracy_ceiling,
        })
    }

    pub fn plan_for_assembly(
        &self,
        request: &StructuralRandomResponseAssemblyRequest,
        inputs: PreparedInputResourceFacts,
        result_output: PreparedOutputResourceFacts,
        receipt_output: PreparedOutputResourceFacts,
        state_profiles_output: PreparedOutputResourceFacts,
        output_set: PreparedOutputSetResourceFacts,
        local_execution: LocalExecutionPlan,
    ) -> Result<FeaResourcePlan, ResourcePlanningError> {
        request.validate().map_err(invalid_error)?;
        request
            .scope()
            .validate_for(&self.workload)
            .map_err(invalid_error)?;
        if request.scope().topology_digest() != self.topology_digest
            || request.scope().basis() != self.basis
            || request.scope().operator_digest() != self.operator_digest
        {
            return Err(invalid(
                "random-response assembly differs from its admitted topology or basis",
            ));
        }
        let task_count = count(request.tasks().len(), "random-response tasks")?;
        let interval_count = count(
            request.scope().plan().interval_count(),
            "random-response intervals",
        )?;
        let interval_encoded_bytes = request.tasks().iter().try_fold(0_u64, |sum, task| {
            add(
                sum,
                count(
                    task.canonical_bytes().map_err(invalid_error)?.len(),
                    "random-response interval task bytes",
                )?,
                "random-response interval result bytes",
            )
        })?;
        let response_value_count = count(
            request.scope().plan().value_count(),
            "random-response response values",
        )?;
        let result_bytes = structural_random_response_assembly_payload_bytes_upper_bound(request)?;
        let receipt_bytes = structural_random_response_receipt_payload_bytes_upper_bound();
        let profiles_bytes =
            crate::execution::structural_state_publication_profiles_encoding_profile(
                self.workload.model(),
                self.workload.procedure_step().map_err(invalid_error)?,
                &[crate::execution::SemanticStatePublicationSource::StructuralRandomStatistics],
                request.scope().topology_digest(),
            )
            .map_err(invalid_error)?
            .encoded_bytes_upper_bound;
        validate_output(
            &state_profiles_output,
            profiles_bytes,
            "random-response publication profiles",
        )?;
        validate_output(&result_output, result_bytes, "random-response result")?;
        validate_output(&receipt_output, receipt_bytes, "random-response receipt")?;
        let ordering_workspace = multiply(
            multiply(
                response_value_count,
                3,
                "random-response total, compensation, and flat vectors",
            )?,
            8,
            "random-response assembly workspace",
        )?;
        build_resource_plan(FeaResourcePlanningInput {
            estimation_model_revision: RESOURCE_ESTIMATION_MODEL_REVISION,
            phase: FeaResourcePhase::StructuralRandomResponseAssembly(request.identity()),
            demand: FeaResourceDemand::StructuralRandomResponseAssembly(
                StructuralRandomResponseAssemblyResourceDemand {
                    scope_digest: request.scope().identity().digest(),
                    task_count: Count(task_count),
                    interval_count: Count(interval_count),
                    interval_encoded_bytes: ByteCount(interval_encoded_bytes),
                    response_value_count: Count(response_value_count),
                    inputs,
                    ordering_workspace_bytes_upper_bound: ByteCount(ordering_workspace),
                    result_output,
                    receipt_output,
                    state_profiles_output,
                },
            ),
            output_set: Some(output_set),
            worker: self.worker(local_execution)?,
            accuracy_ceiling: self.accuracy_ceiling,
        })
    }
}

pub fn structural_random_response_interval_payload_bytes_upper_bound(
    request: &StructuralRandomResponseIntervalRequest,
) -> Result<u64, ResourcePlanningError> {
    request.validate().map_err(invalid_error)?;
    let request_bytes = count(
        request.canonical_bytes().map_err(invalid_error)?.len(),
        "random-response interval request bytes",
    )?;
    let interval_count = count(
        request.interval_range()[1] - request.interval_range()[0],
        "random-response task intervals",
    )?;
    let response_values = count(
        request.scope().plan().value_count(),
        "random-response interval values",
    )?;
    add(
        INTERVAL_FIXED_PAYLOAD_BYTES,
        add(
            multiply(request_bytes, 2, "embedded interval request copies")?,
            multiply(
                interval_count,
                add(
                    INTERVAL_METADATA_BYTES,
                    multiply(response_values, 8, "interval variance bytes")?,
                    "interval contribution bytes",
                )?,
                "random-response interval contributions",
            )?,
            "random-response interval variable payload",
        )?,
        "random-response interval payload",
    )
}

pub fn structural_random_response_assembly_payload_bytes_upper_bound(
    request: &StructuralRandomResponseAssemblyRequest,
) -> Result<u64, ResourcePlanningError> {
    request.validate().map_err(invalid_error)?;
    let request_bytes = count(
        request.canonical_bytes().map_err(invalid_error)?.len(),
        "random-response assembly request bytes",
    )?;
    let response_values = count(
        request.scope().plan().value_count(),
        "random-response assembly values",
    )?;
    add(
        ASSEMBLY_FIXED_PAYLOAD_BYTES,
        add(
            request_bytes,
            multiply(
                multiply(response_values, 2, "variance and RMS values")?,
                8,
                "random-response result value bytes",
            )?,
            "random-response result variable payload",
        )?,
        "random-response result payload",
    )
}

pub const fn structural_random_response_receipt_payload_bytes_upper_bound() -> u64 {
    RECEIPT_PAYLOAD_BYTES
}

fn validate_output(
    output: &PreparedOutputResourceFacts,
    expected: u64,
    label: &'static str,
) -> Result<(), ResourcePlanningError> {
    if output.encoded_payload_bytes_upper_bound.0 != expected
        || output.transport_closure_bytes_upper_bound.0 < expected
        || output.construction_resident_bytes_admission_budget.0 < expected
    {
        return Err(invalid(format!(
            "{label} output facts differ from its canonical payload ceiling"
        )));
    }
    Ok(())
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

fn add(left: u64, right: u64, label: &'static str) -> Result<u64, ResourcePlanningError> {
    left.checked_add(right)
        .ok_or(ResourcePlanningError::Overflow(label))
}

fn multiply(left: u64, right: u64, label: &'static str) -> Result<u64, ResourcePlanningError> {
    left.checked_mul(right)
        .ok_or(ResourcePlanningError::Overflow(label))
}

fn invalid(message: impl Into<String>) -> ResourcePlanningError {
    ResourcePlanningError::Invalid(message.into())
}

fn invalid_error(error: impl ToString) -> ResourcePlanningError {
    invalid(error.to_string())
}
