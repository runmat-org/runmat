use std::collections::BTreeSet;

use runmat_analysis_core::{ModalCombination, ModalTruncationCorrection, SpectrumResponse};
use runmat_execution::resource::AcceleratorRequest;
use runmat_meshing_core::{CanonicalMeshingContract, SolverMeshArtifact};

use crate::{
    algebra::CoercivityWorkEnvelope,
    constraints::CanonicalStructuralConstraintPlan,
    execution::{
        CanonicalSemanticState, StructuralModalResponseSpectrumAssemblyRequest,
        StructuralModalResponseSpectrumDirectionRequest, StructuralModalResponseSpectrumWorkload,
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
    ResourcePlanningError, SolveResourcePolicy,
    StructuralModalResponseSpectrumAssemblyResourceDemand,
    StructuralModalResponseSpectrumDirectionResourceDemand, StructuralOperatorResourceDemand,
    WorkerResourcePolicy, RESOURCE_ESTIMATION_MODEL_REVISION,
};

const FIXED_WORKER_MEMORY_BYTES: u64 = 16 * 1024 * 1024;
const FIXED_WORKER_SCRATCH_BYTES: u64 = 8 * 1024 * 1024;
const FIELD_RECOVERY_FIXED_PAYLOAD_BYTES: u64 = 256 * 1024;

mod sizing;

use sizing::{missing_mass_static_system_workspace_bytes, validate_output};
pub use sizing::{
    structural_modal_response_spectrum_assembly_payload_bytes_upper_bound,
    structural_modal_response_spectrum_direction_payload_bytes_upper_bound,
    structural_modal_response_spectrum_receipt_payload_bytes_upper_bound,
};

/// Reusable exact projection for every independently scheduled direction in one spectrum step.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreparedStructuralResponseSpectrumAssemblyOutputs {
    pub result: PreparedOutputResourceFacts,
    pub receipt: PreparedOutputResourceFacts,
    pub field_recovery: PreparedOutputResourceFacts,
    pub state_profiles: PreparedOutputResourceFacts,
}

/// Reusable exact projection for every independently scheduled direction in one spectrum step.
#[derive(Clone)]
pub struct StructuralModalResponseSpectrumResourceProjection {
    workload: StructuralModalResponseSpectrumWorkload,
    topology_digest: runmat_execution::Digest,
    topology: super::TopologyResourceFacts,
    sparse: super::SparseResourceFacts,
    constraints: super::ConstraintResourceFacts,
    assembly: AssemblyResourcePolicy,
    excitation_workspace: super::ExcitationWorkspaceResourceFacts,
    basis: crate::ModalBasisIdentity,
    operator_digest: runmat_execution::Digest,
    physical_dof_count: u64,
    mode_count: u64,
    linear_field_value_count_upper_bound: u64,
    missing_mass_reduced_dof_count: u64,
    missing_mass_system_workspace_bytes_upper_bound: u64,
    maximum_wall_milliseconds: u64,
    accuracy_ceiling: runmat_analysis_core::AccuracyResourceCeiling,
}

impl StructuralModalResponseSpectrumResourceProjection {
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
        workload: &StructuralModalResponseSpectrumWorkload,
        modal_state: &CanonicalSemanticState,
        accelerator_requests: &[AcceleratorRequest],
    ) -> Result<Self, ResourcePlanningError> {
        workload.validate().map_err(invalid_error)?;
        modal_state.validate().map_err(invalid_error)?;
        mesh.validate_canonical().map_err(invalid_error)?;
        if workload.backend() != ComputeBackend::Cpu || !accelerator_requests.is_empty() {
            return Err(invalid(
                "modal response-spectrum execution currently requires CPU resources without accelerators",
            ));
        }
        let topology_digest = runmat_execution::Digest::from_bytes(*mesh.canonical_digest.bytes());
        let basis = modal_state.payload.modal_basis().ok_or_else(|| {
            invalid("modal response-spectrum resource admission requires a modal-basis state")
        })?;
        if modal_state.identity != workload.modal_basis_transition().source_state
            || modal_state.topology_digest != topology_digest
            || workload.modal_basis_transition().target_topology_digest != topology_digest
        {
            return Err(invalid(
                "modal response-spectrum workload, state, and mesh topology differ",
            ));
        }
        let facts = derive_structural_resource_facts(workload.model(), Some(mesh), constraints)?;
        let physical_dof_count = facts.sparse.rows.0;
        let reduction = facts
            .constraint_matrix
            .affine_reduction()
            .map_err(invalid_error)?;
        if reduction.offset.iter().any(|value| *value != 0.0) || reduction.reduced_dof_count == 0 {
            return Err(invalid(
                "modal response-spectrum resource admission requires a nonempty homogeneous reduction",
            ));
        }
        let reduced_stiffness = reduction
            .reduce_sparse_pattern(facts.symbolic.stiffness_upper_bound())
            .map_err(invalid_error)?;
        let missing_mass_reduced_dof_count = count(
            reduction.reduced_dof_count,
            "missing-mass reduced degree-of-freedom count",
        )?;
        let missing_mass_system_workspace_bytes_upper_bound =
            missing_mass_static_system_workspace_bytes(
                physical_dof_count,
                missing_mass_reduced_dof_count,
                count(
                    reduced_stiffness.column_indices().len(),
                    "missing-mass reduced stiffness nonzeros",
                )?,
                CoercivityWorkEnvelope::derive(&reduced_stiffness)
                    .map_err(invalid_error)?
                    .resident_bytes_limit(),
            )?;
        if basis
            .modes()
            .iter()
            .any(|mode| u64::try_from(mode.dof_shape().len()).ok() != Some(physical_dof_count))
            || basis.residual_vectors().iter().any(|residual| {
                u64::try_from(residual.dof_shape().len()).ok() != Some(physical_dof_count)
            })
        {
            return Err(invalid(
                "modal response-spectrum basis and structural DOF counts differ",
            ));
        }
        for required in workload
            .required_residual_directions()
            .map_err(invalid_error)?
        {
            let expected = normalized(required.vector)?;
            if !basis.residual_vectors().iter().any(|residual| {
                residual
                    .direction()
                    .iter()
                    .zip(expected)
                    .all(|(left, right)| (left - right).abs() <= 1.0e-10)
            }) {
                return Err(invalid(
                    "modal basis omits a residual vector required by the response spectra",
                ));
            }
        }
        let mode_count = count(
            basis.modes().len() + basis.residual_vectors().len(),
            "response-spectrum mode count",
        )?;
        let nodal_values = multiply(
            count(
                facts.recovery_node_count,
                "response-spectrum recovery nodes",
            )?,
            if facts.recovery_has_rotational_dofs {
                6
            } else {
                3
            },
            "response-spectrum nodal field values",
        )?;
        let solid_values = multiply(
            count(
                facts.recovery_solid_element_count,
                "response-spectrum solid recovery elements",
            )?,
            12,
            "response-spectrum solid strain and stress values",
        )?;
        let shell_values = multiply(
            count(
                facts.recovery_shell_element_count,
                "response-spectrum shell recovery elements",
            )?,
            8,
            "response-spectrum shell resultant values",
        )?;
        let linear_field_value_count_upper_bound = add(
            add(
                nodal_values,
                solid_values,
                "response-spectrum nodal and solid fields",
            )?,
            shell_values,
            "response-spectrum linear field values",
        )?;
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
                add(sum, value.0, "operator contributions")
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
                        multiply(contribution_nonzeros, 24, "operator contribution bytes")?,
                        facts.connector_assembly_buffer_bytes,
                    )?,
                ),
                scratch_bytes_per_element: ByteCount(ASSEMBLY_SCRATCH_BYTES_PER_ELEMENT),
                fixed_scratch_bytes_per_worker: ByteCount(FIXED_ASSEMBLY_SCRATCH_BYTES_PER_WORKER),
            },
            basis: basis.identity(),
            operator_digest: basis.operator_digest(),
            physical_dof_count,
            mode_count,
            linear_field_value_count_upper_bound,
            missing_mass_reduced_dof_count,
            missing_mass_system_workspace_bytes_upper_bound,
            maximum_wall_milliseconds: step.accuracy.resources.maximum_wall_time_milliseconds,
            accuracy_ceiling: step.accuracy.resources,
        })
    }

    pub fn plan_for_direction(
        &self,
        request: &StructuralModalResponseSpectrumDirectionRequest,
        inputs: PreparedInputResourceFacts,
        output: PreparedOutputResourceFacts,
        output_set: PreparedOutputSetResourceFacts,
        local_execution: LocalExecutionPlan,
    ) -> Result<FeaResourcePlan, ResourcePlanningError> {
        request
            .validate_for(&self.workload)
            .map_err(invalid_error)?;
        if request.scope().topology_digest() != self.topology_digest
            || request.scope().basis() != self.basis
            || request.scope().operator_digest() != self.operator_digest
            || count(
                request.scope().physical_dof_count(),
                "response-spectrum physical DOFs",
            )? != self.physical_dof_count
        {
            return Err(invalid(
                "response-spectrum request differs from its admitted topology",
            ));
        }
        let point_count = count(
            request.spectrum().abscissa.canonical_points().len(),
            "spectrum points",
        )?;
        let correlation_pairs = if matches!(
            request.spectrum().modal_combination,
            ModalCombination::CompleteQuadraticCombination | ModalCombination::TenPercent
        ) {
            multiply(self.mode_count, self.mode_count, "modal correlation pairs")?
        } else {
            self.mode_count
        };
        let response_vector_count = add(self.mode_count, 8, "response work vectors")?;
        let workspace_values = add(
            multiply(
                self.physical_dof_count,
                response_vector_count,
                "response work vector values",
            )?,
            add(
                correlation_pairs,
                multiply(point_count, 2, "interpolation values")?,
                "correlation and interpolation",
            )?,
            "response-spectrum workspace values",
        )?;
        let workspace_bytes = multiply(workspace_values, 8, "response-spectrum workspace bytes")?;
        let result_bytes =
            structural_modal_response_spectrum_direction_payload_bytes_upper_bound(request)?;
        validate_output(&output, result_bytes, "response-spectrum direction")?;
        let artifact_overhead = output
            .transport_closure_bytes_upper_bound
            .0
            .checked_sub(result_bytes)
            .ok_or_else(|| invalid("response-spectrum direction transport is below its payload"))?;
        build_resource_plan(FeaResourcePlanningInput {
            estimation_model_revision: RESOURCE_ESTIMATION_MODEL_REVISION,
            phase: FeaResourcePhase::StructuralModalResponseSpectrumDirection(request.identity()),
            demand: FeaResourceDemand::StructuralModalResponseSpectrumDirection(
                StructuralModalResponseSpectrumDirectionResourceDemand {
                    operator: StructuralOperatorResourceDemand {
                        excitation_workspace: self.excitation_workspace.clone(),
                        topology_digest: self.topology_digest,
                        numerical_scope_digest: request.scope().operator_digest(),
                        topology: self.topology.clone(),
                        sparse: self.sparse.clone(),
                        constraints: self.constraints.clone(),
                        assembly: self.assembly.clone(),
                        solve: SolveResourcePolicy::NoSolve {
                            workspace_bytes_upper_bound: ByteCount(workspace_bytes),
                        },
                        phase_resources: PhaseResourcePolicy {
                            resident_value_count: Count(workspace_values),
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
                            samples: Count(1),
                        },
                    },
                    spectrum_point_count: Count(point_count),
                    mode_count_upper_bound: Count(self.mode_count),
                    correlation_pair_count_upper_bound: Count(correlation_pairs),
                    response_vector_count_upper_bound: Count(response_vector_count),
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
        request: &StructuralModalResponseSpectrumAssemblyRequest,
        inputs: PreparedInputResourceFacts,
        outputs: PreparedStructuralResponseSpectrumAssemblyOutputs,
        output_set: PreparedOutputSetResourceFacts,
        local_execution: LocalExecutionPlan,
    ) -> Result<FeaResourcePlan, ResourcePlanningError> {
        let PreparedStructuralResponseSpectrumAssemblyOutputs {
            result: result_output,
            receipt: receipt_output,
            field_recovery: field_recovery_output,
            state_profiles: state_profiles_output,
        } = outputs;
        request.validate().map_err(invalid_error)?;
        request
            .scope()
            .validate_for(&self.workload)
            .map_err(invalid_error)?;
        let direction_count = count(request.entries().len(), "response-spectrum directions")?;
        let direction_bytes = request.entries().iter().try_fold(0_u64, |sum, entry| {
            add(
                sum,
                count(
                    entry.canonical_bytes().map_err(invalid_error)?.len(),
                    "direction bytes",
                )?,
                "direction result bytes",
            )
        })?;
        let result_bytes =
            structural_modal_response_spectrum_assembly_payload_bytes_upper_bound(request)?;
        let receipt_bytes = structural_modal_response_spectrum_receipt_payload_bytes_upper_bound();
        let profiles_bytes =
            crate::execution::structural_state_publication_profiles_encoding_profile(
                self.workload.model(),
                self.workload.procedure_step().map_err(invalid_error)?,
                &[crate::execution::SemanticStatePublicationSource::StructuralResponseEnvelope],
                request.scope().topology_digest(),
            )
            .map_err(invalid_error)?
            .encoded_bytes_upper_bound;
        validate_output(
            &state_profiles_output,
            profiles_bytes,
            "response-spectrum publication profiles",
        )?;
        let field_recovery_bytes = self.field_recovery_payload_bytes_upper_bound()?;
        validate_output(&result_output, result_bytes, "response-spectrum result")?;
        validate_output(&receipt_output, receipt_bytes, "response-spectrum receipt")?;
        validate_output(
            &field_recovery_output,
            field_recovery_bytes,
            "response-spectrum field recovery",
        )?;
        let ordering_workspace = multiply(
            add(
                multiply(
                    self.physical_dof_count,
                    direction_count,
                    "direction vectors",
                )?,
                self.physical_dof_count,
                "combined response vector",
            )?,
            8,
            "response-spectrum assembly workspace",
        )?;
        let field_recovery_workspace = multiply(
            multiply(
                self.linear_field_value_count_upper_bound,
                self.mode_count,
                "response-spectrum modal field values",
            )?,
            8,
            "response-spectrum field recovery workspace",
        )?;
        let missing_mass_static_solve_count = count(
            request
                .entries()
                .iter()
                .filter(|entry| {
                    let spectrum = entry.request().spectrum();
                    spectrum.response == SpectrumResponse::PseudoAcceleration
                        && matches!(
                            spectrum.truncation.correction,
                            ModalTruncationCorrection::MissingMass { .. }
                        )
                })
                .count(),
            "missing-mass static solve count",
        )?;
        let (
            missing_mass_reduced_dof_count,
            missing_mass_system_workspace_bytes_upper_bound,
            missing_mass_total_iteration_count_upper_bound,
        ) = if missing_mass_static_solve_count == 0 {
            (0, 0, 0)
        } else {
            let iteration_count_per_solve = count(
                crate::pipeline::homogeneous_static::homogeneous_static_iteration_limit(
                    usize::try_from(self.missing_mass_reduced_dof_count).map_err(|_| {
                        ResourcePlanningError::Overflow("missing-mass reduced dimension")
                    })?,
                ),
                "missing-mass static iteration limit",
            )?;
            (
                self.missing_mass_reduced_dof_count,
                self.missing_mass_system_workspace_bytes_upper_bound,
                multiply(
                    missing_mass_static_solve_count,
                    iteration_count_per_solve,
                    "missing-mass total static iterations",
                )?,
            )
        };
        build_resource_plan(FeaResourcePlanningInput {
            estimation_model_revision: RESOURCE_ESTIMATION_MODEL_REVISION,
            phase: FeaResourcePhase::StructuralModalResponseSpectrumAssembly(request.identity()),
            demand: FeaResourceDemand::StructuralModalResponseSpectrumAssembly(
                StructuralModalResponseSpectrumAssemblyResourceDemand {
                    scope_digest: request.scope().identity().digest(),
                    direction_count: Count(direction_count),
                    direction_encoded_bytes: ByteCount(direction_bytes),
                    physical_dof_count: Count(self.physical_dof_count),
                    inputs,
                    ordering_workspace_bytes_upper_bound: ByteCount(ordering_workspace),
                    field_recovery_workspace_bytes_upper_bound: ByteCount(field_recovery_workspace),
                    missing_mass_static_solve_count: Count(missing_mass_static_solve_count),
                    missing_mass_reduced_dof_count: Count(missing_mass_reduced_dof_count),
                    missing_mass_system_workspace_bytes_upper_bound: ByteCount(
                        missing_mass_system_workspace_bytes_upper_bound,
                    ),
                    missing_mass_total_iteration_count_upper_bound: Count(
                        missing_mass_total_iteration_count_upper_bound,
                    ),
                    result_output,
                    receipt_output,
                    field_recovery_output,
                    state_profiles_output,
                },
            ),
            output_set: Some(output_set),
            worker: self.worker(local_execution)?,
            accuracy_ceiling: self.accuracy_ceiling,
        })
    }

    pub fn field_recovery_payload_bytes_upper_bound(&self) -> Result<u64, ResourcePlanningError> {
        add(
            FIELD_RECOVERY_FIXED_PAYLOAD_BYTES,
            multiply(
                self.linear_field_value_count_upper_bound,
                8,
                "response-spectrum linear field payload",
            )?,
            "response-spectrum field recovery payload",
        )
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

fn normalized(vector: [f64; 3]) -> Result<[f64; 3], ResourcePlanningError> {
    let norm = vector.iter().map(|value| value * value).sum::<f64>().sqrt();
    if !norm.is_finite() || norm <= f64::EPSILON {
        return Err(invalid(
            "response-spectrum direction is not finite and nonzero",
        ));
    }
    Ok(vector.map(|value| value / norm))
}
