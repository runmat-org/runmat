//! Assembly-free resource projection for constructing a transient checkpoint scope.

use std::collections::BTreeSet;

use runmat_execution::Digest;
use runmat_meshing_core::{CanonicalMeshingContract, SolverMeshArtifact};

use crate::{
    constraints::CanonicalStructuralConstraintPlan,
    execution::{
        AnalysisWorkKind, AnalysisWorkloadIdentity, CanonicalSemanticState, ExecutionPlanIdentity,
        SequenceIdentity, StepIdentity, StructuralTransientSolverPlan, StructuralTransientWorkload,
        StudyIdentity,
    },
    pipeline::transient::PreparedTransientPreassembly,
};

use super::{
    assembly_sizing::{
        ASSEMBLY_SCRATCH_BYTES_PER_ELEMENT, FIXED_ASSEMBLY_SCRATCH_BYTES_PER_WORKER,
    },
    build_resource_plan,
    structural_facts::derive_structural_resource_facts,
    AccuracyResourceDemand, AssemblyResourcePolicy, ByteCount, CheckpointArtifactPolicy, Count,
    FeaResourceDemand, FeaResourcePhase, FeaResourcePlan, FeaResourcePlanningInput,
    FieldRecoveryResourceFacts, ObservationResourceFacts, PartitioningPolicy, PhaseResourcePolicy,
    PreparedInputResourceFacts, PreparedOutputResourceFacts, PreparedOutputSetResourceFacts,
    ResourcePlanningError, SolveResourcePolicy, StructuralOperatorResourceDemand,
    StructuralTransientScopeResourceDemand, WorkerResourcePolicy,
    RESOURCE_ESTIMATION_MODEL_REVISION,
};

const FIXED_MEMORY_BYTES: u64 = 16 * 1024 * 1024;
const FIXED_SCRATCH_BYTES: u64 = 8 * 1024 * 1024;
// One point has only a f64 and two closed enum tags; 128 bytes covers its canonical map keys,
// CBOR headers, and values without allocating an encoded schedule during admission.
const SCOPE_BYTES_PER_SCHEDULE_POINT_UPPER_BOUND: u64 = 128;
// Every scope field outside `schedule` is a fixed-size identity, scalar, or the bounded modal
// projection evidence. Reserve its canonical map/field encoding independently of sample count.
const SCOPE_FIXED_PAYLOAD_BYTES_UPPER_BOUND: u64 = 16 * 1024;

pub(super) fn transient_source_workload_identity(
    workload: &StructuralTransientWorkload,
    mesh: &SolverMeshArtifact,
    execution_plan: ExecutionPlanIdentity,
) -> Result<(StudyIdentity, StepIdentity, AnalysisWorkloadIdentity), ResourcePlanningError> {
    let study = StudyIdentity::from_authored_contract(&workload.model, None)
        .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?;
    let sequence = SequenceIdentity::from_solver_contract(&workload.model.sequence)
        .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?;
    let step = StepIdentity::from_solver_contract(
        sequence,
        workload
            .procedure_step()
            .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?,
    )
    .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?;
    let workload_bytes = workload
        .canonical_bytes()
        .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?;
    let mesh_digest = Digest::from_bytes(*mesh.canonical_digest.bytes());
    let mut dependencies = vec![mesh_digest];
    if let StructuralTransientSolverPlan::ModalReduced { basis_transition } = &workload.solver {
        dependencies.extend([
            basis_transition.source_state.digest(),
            basis_transition.identity.digest(),
        ]);
    }
    let source = AnalysisWorkloadIdentity::new(
        study,
        step,
        execution_plan,
        AnalysisWorkKind::SolveStructuralTransientBlock,
        Digest::sha256(&workload_bytes),
        dependencies,
    )
    .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?;
    Ok((study, step, source))
}

pub(crate) fn structural_transient_scope_payload_bytes_upper_bound(
    schedule: &crate::execution::StructuralTransientSchedule,
) -> Result<u64, ResourcePlanningError> {
    schedule
        .validate()
        .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?;
    add(
        multiply(
            count(schedule.points.len(), "transient schedule points")?,
            SCOPE_BYTES_PER_SCHEDULE_POINT_UPPER_BOUND,
            "transient schedule payload bytes",
        )?,
        SCOPE_FIXED_PAYLOAD_BYTES_UPPER_BOUND,
        "transient scope payload bytes",
    )
}

pub(crate) fn build_structural_transient_scope_resource_plan(
    workload: &StructuralTransientWorkload,
    mesh: &SolverMeshArtifact,
    modal_state: Option<&CanonicalSemanticState>,
    prepared: &PreparedTransientPreassembly,
    prepared_inputs: Option<PreparedInputResourceFacts>,
    output_set: Option<PreparedOutputSetResourceFacts>,
    local_execution: super::LocalExecutionPlan,
) -> Result<FeaResourcePlan, ResourcePlanningError> {
    workload
        .validate()
        .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?;
    mesh.validate_canonical()
        .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?;
    if prepared.execution_step.step_id != workload.step_id {
        return Err(ResourcePlanningError::Invalid(
            "transient preassembly step differs from the workload".into(),
        ));
    }
    let workload_bytes = workload
        .canonical_bytes()
        .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?;
    let mesh_bytes = mesh
        .canonical_encode()
        .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?;
    let source_workload_digest = Digest::sha256(&workload_bytes);
    let constraints =
        CanonicalStructuralConstraintPlan::compile_for_model(&prepared.active_model, Some(mesh))
            .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?;
    let facts = derive_structural_resource_facts(&prepared.active_model, Some(mesh), &constraints)?;
    crate::execution::validate_initial_field_selection(
        &prepared.active_model,
        &prepared.execution_step,
        facts.topology_digest,
        usize::try_from(facts.topology.nodes.0)
            .map_err(|_| ResourcePlanningError::Overflow("initial-field node count"))?,
        &workload.initial_fields,
    )
    .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?;
    let dofs = facts.sparse.rows.0;
    let operator_nonzeros = facts
        .sparse
        .operator_nonzeros
        .iter()
        .try_fold(0_u64, |sum, count| {
            add(sum, count.0, "transient operator nonzeros")
        })?;
    let operator_storage_bytes = add(
        multiply(operator_nonzeros, 16, "transient CSR entries")?,
        multiply(
            add(dofs, 1, "transient CSR rows")?,
            multiply(
                u64::try_from(facts.sparse.operator_nonzeros.len())
                    .map_err(|_| ResourcePlanningError::Overflow("transient operator count"))?,
                8,
                "transient CSR row bytes",
            )?,
            "transient CSR row offsets",
        )?,
        "transient CSR storage",
    )?;
    let modal_values = match (&workload.solver, modal_state) {
        (StructuralTransientSolverPlan::Direct, None) => 0,
        (StructuralTransientSolverPlan::ModalReduced { basis_transition }, Some(state)) => {
            state
                .validate()
                .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?;
            let basis = state.payload.modal_basis().ok_or_else(|| {
                ResourcePlanningError::Invalid(
                    "modal transient scope requires a modal basis".into(),
                )
            })?;
            if state.identity != basis_transition.source_state
                || state.topology_digest != Digest::from_bytes(*mesh.canonical_digest.bytes())
            {
                return Err(ResourcePlanningError::Invalid(
                    "modal transient scope basis identity or topology differs".into(),
                ));
            }
            let modes = u64::try_from(basis.modes().len())
                .map_err(|_| ResourcePlanningError::Overflow("modal basis modes"))?;
            add(
                multiply(
                    multiply(dofs, modes, "modal shape values")?,
                    4,
                    "modal shape workspace",
                )?,
                multiply(
                    multiply(modes, modes, "modal dense values")?,
                    11,
                    "modal dense workspace",
                )?,
                "modal projection values",
            )?
        }
        _ => {
            return Err(ResourcePlanningError::Invalid(
                "transient solver and modal-basis presence differ".into(),
            ));
        }
    };
    let modal_bytes = multiply(modal_values, 8, "modal projection bytes")?;
    let modal_bytes = add(
        modal_bytes,
        multiply(dofs, 14 * 8, "modal full-order vectors")?,
        "modal projection workspace",
    )?;
    let input_bytes = add(
        count(workload_bytes.len(), "transient workload bytes")?,
        count(mesh_bytes.len(), "transient mesh bytes")?,
        "transient inputs",
    )?;
    let modal_input_bytes = modal_state
        .map(|state| {
            state
                .canonical_bytes()
                .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))
                .and_then(|bytes| count(bytes.len(), "modal-state bytes"))
        })
        .transpose()?
        .unwrap_or(0);
    let encoded_inputs = add(input_bytes, modal_input_bytes, "scope encoded inputs")?;
    let decoded_inputs =
        runmat_canonical_codec::projected_canonical_typed_decode_resident_bytes_estimate(
            encoded_inputs,
            3,
        )
        .ok_or(ResourcePlanningError::Overflow("scope decoded inputs"))?;
    // Scope construction retains both the authored model and its active-constraint copy.
    // The projected materialization allowance also covers motion objects and mesh clones.
    let inputs = match prepared_inputs {
        Some(facts)
            if facts.encoded_payload_bytes.0 == encoded_inputs
                && facts.transport_closure_bytes.0 >= encoded_inputs
                && facts.decoded_resident_bytes_admission_budget.0 >= encoded_inputs =>
        {
            facts
        }
        Some(_) => {
            return Err(ResourcePlanningError::Invalid(
                "prepared transient scope input facts differ from canonical inputs".into(),
            ))
        }
        None => PreparedInputResourceFacts {
            encoded_payload_bytes: ByteCount(encoded_inputs),
            transport_closure_bytes: ByteCount(encoded_inputs),
            decoded_resident_bytes_admission_budget: ByteCount(decoded_inputs),
        },
    };
    let schedule_points = count(prepared.schedule.points.len(), "transient schedule points")?;
    let schedule_bytes = multiply(schedule_points, 128, "transient schedule residency")?;
    let scope_payload_bytes =
        structural_transient_scope_payload_bytes_upper_bound(&prepared.schedule)?;
    let scope_transport_bytes = add(
        scope_payload_bytes,
        crate::execution::ANALYSIS_MANIFEST_MAX_ENCODED_BYTES,
        "transient scope output closure",
    )?;
    let scope_construction_bytes =
        runmat_canonical_codec::projected_canonical_encoding_resident_bytes_estimate(
            scope_payload_bytes,
            1,
        )
        .and_then(|bytes| bytes.checked_add(scope_transport_bytes))
        .ok_or(ResourcePlanningError::Overflow(
            "transient scope output construction",
        ))?;
    let scope_output = PreparedOutputResourceFacts {
        encoded_payload_bytes_upper_bound: ByteCount(scope_payload_bytes),
        transport_closure_bytes_upper_bound: ByteCount(scope_transport_bytes),
        construction_resident_bytes_admission_budget: ByteCount(scope_construction_bytes),
    };
    // The output set is the one root manifest listing this phase's output roots. Its bound is
    // the root schema bound, not the numerical payload bound carried by the prepared output.
    let output_set = output_set.unwrap_or(PreparedOutputSetResourceFacts {
        encoded_bytes_upper_bound: ByteCount(
            crate::execution::ANALYSIS_OUTPUT_SET_MAX_ENCODED_BYTES,
        ),
    });
    if output_set.encoded_bytes_upper_bound.0 == 0
        || output_set.encoded_bytes_upper_bound.0
            > crate::execution::ANALYSIS_OUTPUT_SET_MAX_ENCODED_BYTES
    {
        return Err(ResourcePlanningError::Invalid(
            "transient scope output set is outside the output-set root schema bound".into(),
        ));
    }
    let phase_bytes = add(
        add(
            schedule_bytes,
            multiply(dofs, 8 * 6, "scope state and load vectors")?,
            "scope phase state residency",
        )?,
        scope_construction_bytes,
        "scope phase residency",
    )?;
    let scope_workspace = add(
        multiply(
            operator_storage_bytes,
            2,
            "operator identity encoding workspace",
        )?,
        modal_bytes,
        "scope workspace",
    )?;
    // Scope assembly constructs all three operators, so the contribution count must cover
    // their combined entries rather than only the largest individual CSR operator.
    let contribution_nonzeros = operator_nonzeros;
    let contribution_bytes = super::connector_buffers::include_connector_buffers(
        multiply(operator_nonzeros, 24, "assembly contributions")?,
        facts.connector_assembly_buffer_bytes,
    )?;
    build_resource_plan(FeaResourcePlanningInput {
        estimation_model_revision: RESOURCE_ESTIMATION_MODEL_REVISION,
        phase: FeaResourcePhase::StructuralTransientScopePreparation(source_workload_digest),
        demand: FeaResourceDemand::StructuralTransientScopePreparation(
            StructuralTransientScopeResourceDemand {
                source_workload_digest,
                operator: StructuralOperatorResourceDemand {
                    excitation_workspace: super::excitation_workspace::derive_excitation_workspace(
                        &prepared.active_model,
                        &prepared.execution_step,
                        Some(mesh),
                    )?,
                    topology_digest: facts.topology_digest,
                    numerical_scope_digest: facts.symbolic.identity(),
                    topology: facts.topology,
                    sparse: facts.sparse,
                    constraints: facts.constraints,
                    assembly: AssemblyResourcePolicy {
                        contribution_nonzero_upper_bound: Count(contribution_nonzeros),
                        contribution_bytes_upper_bound: ByteCount(contribution_bytes),
                        scratch_bytes_per_element: ByteCount(ASSEMBLY_SCRATCH_BYTES_PER_ELEMENT),
                        fixed_scratch_bytes_per_worker: ByteCount(
                            FIXED_ASSEMBLY_SCRATCH_BYTES_PER_WORKER,
                        ),
                    },
                    solve: SolveResourcePolicy::NoSolve {
                        workspace_bytes_upper_bound: ByteCount(scope_workspace),
                    },
                    phase_resources: PhaseResourcePolicy {
                        resident_value_count: Count(multiply(dofs, 6, "scope resident values")?),
                        resident_bytes_upper_bound: ByteCount(phase_bytes),
                        scratch_bytes: ByteCount(scope_workspace),
                        retained_bytes: ByteCount(schedule_bytes),
                    },
                    field_recovery: FieldRecoveryResourceFacts::default(),
                    observations: ObservationResourceFacts {
                        maximum_samples_per_observation: Count(0),
                        produced_sample_count: Count(0),
                        retained_sample_count: Count(0),
                        operation_count: Count(0),
                        peak_resident_bytes: ByteCount(0),
                        stored_bytes: ByteCount(0),
                        transfer_bytes: ByteCount(0),
                        collective_count: Count(0),
                    },
                    publication: CheckpointArtifactPolicy {
                        // The scope payload and its manifest; the output-set root is separate.
                        result_artifact_bytes: ByteCount(scope_transport_bytes),
                        checkpoint_bytes: ByteCount(0),
                        artifact_overhead_bytes: ByteCount(0),
                        relay_bytes: ByteCount(0),
                        result_egress_bytes: ByteCount(scope_transport_bytes),
                        checkpoint_egress_bytes: ByteCount(0),
                    },
                    partitioning: PartitioningPolicy::Monolithic,
                    accuracy_demand: AccuracyResourceDemand {
                        refinement_cycles: Count(0),
                        solver_tightenings: Count(0),
                        samples: Count(schedule_points),
                    },
                },
                inputs,
                scope_output,
            },
        ),
        output_set: Some(output_set),
        worker: WorkerResourcePolicy {
            cpu_millicores: local_execution
                .checked_cpu_millicores()
                .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?,
            maximum_wall_milliseconds: prepared
                .execution_step
                .accuracy
                .resources
                .maximum_wall_time_milliseconds,
            fixed_memory_bytes: ByteCount(FIXED_MEMORY_BYTES),
            fixed_scratch_bytes: ByteCount(FIXED_SCRATCH_BYTES),
            required_capabilities: BTreeSet::new(),
            accelerators: Vec::new(),
        },
        accuracy_ceiling: prepared.execution_step.accuracy.resources,
    })
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
