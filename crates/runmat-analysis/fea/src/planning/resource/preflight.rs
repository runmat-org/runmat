use crate::{
    algebra::CoercivityWorkEnvelope,
    constraints::{
        AffineConstraintReduction, CanonicalStructuralConstraintPlan, ConstraintEnforcement,
    },
    execution::{FieldRecoveryPlan, StructuralLinearWorkload},
    results::{
        field::StructuralRecoveredFieldTopology,
        observation::{ObservationExecutionPolicy, ResolvedObservationPlan},
    },
    solve::preconditioner::SpdPreconditionerKind,
};
use runmat_analysis_core::{AnalysisModel, ProcedureStep};
use runmat_execution::resource::AcceleratorRequest;
use runmat_meshing_core::SolverMeshArtifact;

mod backend;
mod compilation;
pub use compilation::build_structural_resource_plan_from_compilation;
mod linearization;
mod solve_workspace;
pub use backend::require_executable_static_backend;
pub(super) use solve_workspace::saddle_gmres_workspace_bytes;
use solve_workspace::{affine_transpose_index_peak_bytes, solve_policy};

use super::{
    assembly_sizing::{
        ASSEMBLY_SCRATCH_BYTES_PER_ELEMENT, FIXED_ASSEMBLY_SCRATCH_BYTES_PER_WORKER,
    },
    build_resource_plan,
    structural_facts::{derive_structural_resource_facts, StructuralResourceFacts},
    AccuracyResourceDemand, AssemblyResourcePolicy, ByteCount, CheckpointArtifactPolicy, Count,
    FeaResourceDemand, FeaResourcePhase, FeaResourcePlan, FeaResourcePlanningInput,
    FieldRecoveryResourceFacts, LocalExecutionPlan, NumericalScalar, ObservationResourceFacts,
    PartitioningPolicy, PhaseResourcePolicy, PreconditionerStorage, PreparedInputResourceFacts,
    ResourcePlanningError, SolveResourcePolicy, SolverAdmissionResourcePolicy, StaticSolvePath,
    StaticSolveSelection, StructuralLinearPipelineResourceDemand, StructuralOperatorResourceDemand,
    WorkerResourcePolicy, RESOURCE_ESTIMATION_MODEL_REVISION,
};

use super::RESIDENT_MEMORY_RESOURCE;

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
/// observation counts. It validates original bounded compilation before physical assembly;
/// callers with a fresh construction carrier can reuse it through the companion entry point.
/// Portable callers must independently derive `prepared_inputs` from the authenticated input
/// closure, excluding the resource-plan object itself. `None` is for direct numerical execution;
/// these transport reservations supplement, rather than replace, thermal alignment/work storage.
pub fn build_structural_resource_plan(
    mesh: Option<&SolverMeshArtifact>,
    constraints: &CanonicalStructuralConstraintPlan,
    observations: &ResolvedObservationPlan,
    workload: &StructuralLinearWorkload,
    accelerator_requests: &[AcceleratorRequest],
    local_execution: LocalExecutionPlan,
    prepared_inputs: Option<&PreparedInputResourceFacts>,
) -> Result<FeaResourcePlan, ResourcePlanningError> {
    workload
        .validate()
        .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?;
    let step = workload
        .procedure_step()
        .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?;
    build_static_resource_plan(StaticResourcePreflight {
        compilation: None,
        model: &workload.model,
        step,
        backend: workload.backend,
        preconditioner: workload.preconditioner,
        observation_execution: workload.observation_execution,
        mesh,
        constraints,
        observations,
        accelerator_requests,
        local_execution,
        prepared_inputs,
    })
}

/// The shared numerical preflight for portable and direct static execution. The portable
/// entry point additionally validates the workload's sequencing and replay contract.
#[derive(Clone, Copy)]
pub(crate) struct StaticResourcePreflight<'a> {
    pub model: &'a AnalysisModel,
    pub step: &'a ProcedureStep,
    pub backend: crate::ComputeBackend,
    pub preconditioner: SpdPreconditionerKind,
    pub observation_execution: ObservationExecutionPolicy,
    pub mesh: Option<&'a SolverMeshArtifact>,
    pub constraints: &'a CanonicalStructuralConstraintPlan,
    pub compilation: Option<&'a crate::constraints::ConstraintCompilation<'a>>,
    pub observations: &'a ResolvedObservationPlan,
    pub accelerator_requests: &'a [AcceleratorRequest],
    pub local_execution: LocalExecutionPlan,
    pub prepared_inputs: Option<&'a PreparedInputResourceFacts>,
}

/// The static solve path a structural plan recorded (`None` for other plans).
pub fn recorded_static_solve_path(plan: &FeaResourcePlan) -> Option<StaticSolvePath> {
    match &plan.input.demand {
        FeaResourceDemand::StructuralLinearPipeline(demand) => {
            demand.operator.solve.static_solve_path()
        }
        _ => None,
    }
}

/// Plans within the authored ceiling alone (no machine memory is known).
pub(crate) fn build_static_resource_plan(
    input: StaticResourcePreflight<'_>,
) -> Result<FeaResourcePlan, ResourcePlanningError> {
    build_static_resource_plan_selecting(
        input,
        StaticSolveSelection::WithinCeiling {
            machine_memory_bytes: None,
        },
    )
}

pub(crate) fn build_static_resource_plan_selecting(
    input: StaticResourcePreflight<'_>,
    selection: StaticSolveSelection,
) -> Result<FeaResourcePlan, ResourcePlanningError> {
    if let StaticSolveSelection::Recorded(path) = selection {
        validate_static_path(&input, path)?;
    }
    // These original-input facts are independent of the candidate solve path.
    // Retain their symbolic storage once while each candidate checks its own
    // resource ceilings, identity and solver requirements.
    let compiler_peak = compilation::original_peak(&input)?;
    let facts = derive_structural_resource_facts(input.model, input.mesh, input.constraints)?;
    match selection {
        StaticSolveSelection::Recorded(path) => build_resource_plan(
            build_static_resource_input_with_facts(input, path, compiler_peak, &facts)?,
        ),
        StaticSolveSelection::WithinCeiling {
            machine_memory_bytes,
        } => plan_static_solve_within_ceiling(input, machine_memory_bytes, compiler_peak, &facts),
    }
}

/// The proving factorization when its plan fits the effective ceiling; otherwise the elastic
/// rigidity path for a model it admits, whose plan must fit the same ceiling. An explicit AMG
/// request takes the elastic path whenever the model admits it (the factor is never needed);
/// direct, Jacobi and ILU requests keep the factor proof.
///
/// The effective ceiling is min(authored, machine), where the machine figure is the resident
/// memory the placement grants one worker. A plan is admitted when both its resident memory and
/// its scratch fit that ceiling, the limits the worker pool enforces. When neither path fits,
/// the error names both paths' requirements and the ceiling.
fn plan_static_solve_within_ceiling(
    input: StaticResourcePreflight<'_>,
    machine_memory_bytes: Option<u64>,
    compiler_peak: ByteCount,
    facts: &StructuralResourceFacts,
) -> Result<FeaResourcePlan, ResourcePlanningError> {
    let plan_path = |path: StaticSolvePath| {
        build_resource_plan(build_static_resource_input_with_facts(
            input,
            path,
            compiler_peak,
            facts,
        )?)
    };
    if !elastic_rigidity_admits(&input) {
        return plan_path(StaticSolvePath::ProvingFactorization);
    }
    match input.preconditioner {
        SpdPreconditionerKind::Amg => return plan_path(StaticSolvePath::ElasticRigidityMultigrid),
        SpdPreconditionerKind::Auto => {}
        SpdPreconditionerKind::Direct
        | SpdPreconditionerKind::Jacobi
        | SpdPreconditionerKind::Ilu0 => return plan_path(StaticSolvePath::ProvingFactorization),
    }
    let authored = input.step.accuracy.resources.resident_byte_budget();
    let ceiling = machine_memory_bytes.map_or(authored, |machine| machine.min(authored));
    let fits = |plan: &FeaResourcePlan| {
        plan.request.memory_bytes <= ceiling && plan.request.scratch_bytes <= ceiling
    };
    let factorization_resident_bytes = match plan_path(StaticSolvePath::ProvingFactorization) {
        Ok(plan) if fits(&plan) => return Ok(plan),
        Ok(plan) => Some(plan.request.memory_bytes),
        Err(ResourcePlanningError::AccuracyCeiling {
            resource: RESIDENT_MEMORY_RESOURCE,
            required,
            ..
        }) => Some(required),
        // Another authored limit (samples, cycles) binds the factor plan; the elastic plan
        // answers to the same limit and reports it below if it binds there too.
        Err(ResourcePlanningError::AccuracyCeiling { .. }) => None,
        Err(error) => return Err(error),
    };
    let (multigrid_resident_bytes, multigrid_scratch_bytes) =
        match plan_path(StaticSolvePath::ElasticRigidityMultigrid) {
            Ok(plan) if fits(&plan) => return Ok(plan),
            Ok(plan) => (plan.request.memory_bytes, Some(plan.request.scratch_bytes)),
            Err(ResourcePlanningError::AccuracyCeiling {
                resource: RESIDENT_MEMORY_RESOURCE,
                required,
                ..
            }) => (required, None),
            Err(error) => return Err(error),
        };
    Err(ResourcePlanningError::StaticSolveCeiling {
        factorization_resident_bytes,
        multigrid_resident_bytes,
        multigrid_scratch_bytes,
        maximum: ceiling,
    })
}

/// The elastic rigidity proof covers a linear static solid continuum: a solver mesh of
/// tetrahedra, no connectors, couplings or explicit structural elements, and constraints that
/// only fix or keep whole DOFs.
fn elastic_rigidity_admits(input: &StaticResourcePreflight<'_>) -> bool {
    let Some(mesh) = input.mesh else {
        return false;
    };
    input.step.procedure == runmat_analysis_core::AnalysisProcedure::Static
        && input.step.state_inputs.is_empty()
        && crate::assembly::solver_solid::SolidEnergyPreparation::supports_model(input.model)
        && matches!(
            input.constraints.enforcement,
            ConstraintEnforcement::DirectElimination
        )
        && !mesh.topology.volume_elements.is_empty()
        && mesh
            .topology
            .volume_elements
            .iter()
            .all(|element| matches!(element.node_ids.len(), 4 | 10))
}

/// Inventory of the elastic rigidity proof: retained capture bytes and scalar operations.
fn elastic_rigidity_admission(
    mesh: &SolverMeshArtifact,
    dimension: u64,
) -> SolverAdmissionResourcePolicy {
    let widths = || {
        mesh.topology
            .volume_elements
            .iter()
            .map(|element| element.node_ids.len())
    };
    let operations = widths().fold(0_u64, |sum, w| {
        sum.saturating_add((9 * w * w * 40 + w * 64) as u64)
    });
    SolverAdmissionResourcePolicy {
        dimension: Count(dimension),
        factor_nonzero_upper_bound: Count(0),
        update_count_upper_bound: Count(operations),
        resident_bytes_upper_bound: ByteCount(
            crate::algebra::elastic_rigidity::ElasticRigidityCapture::resident_bytes_for(
                mesh.topology.nodes.len() as u64,
                widths(),
            ),
        ),
    }
}

/// The factor that proves coercivity is reserved in full: the topology-derived envelope bounds
/// the exact factorization of any numerical support of the pattern.
pub(super) fn factor_admission(
    pattern: &crate::algebra::SparsePattern,
) -> Result<SolverAdmissionResourcePolicy, ResourcePlanningError> {
    let bound = CoercivityWorkEnvelope::derive(pattern)
        .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?;
    Ok(SolverAdmissionResourcePolicy {
        dimension: Count(bound.dimension()),
        factor_nonzero_upper_bound: Count(bound.factor_nonzero_limit()),
        update_count_upper_bound: Count(bound.factor_flop_limit()),
        // The factor's values, column indices and row offsets are resident while it proves
        // and solves; the solver's own peak estimate is never allowed below that payload.
        resident_bytes_upper_bound: ByteCount(
            bound
                .factor_nonzero_limit()
                .checked_mul(16)
                .and_then(|bytes| bytes.checked_add((bound.dimension() + 1) * 8))
                .ok_or(ResourcePlanningError::Overflow("static factor payload"))?
                .max(bound.factor_resident_limit()),
        ),
    })
}

pub(super) fn build_static_resource_input(
    input: StaticResourcePreflight<'_>,
    path: StaticSolvePath,
) -> Result<FeaResourcePlanningInput, ResourcePlanningError> {
    validate_static_path(&input, path)?;
    let compiler_peak = compilation::original_peak(&input)?;
    let facts = derive_structural_resource_facts(input.model, input.mesh, input.constraints)?;
    build_static_resource_input_with_facts(input, path, compiler_peak, &facts)
}

fn validate_static_path(
    input: &StaticResourcePreflight<'_>,
    path: StaticSolvePath,
) -> Result<(), ResourcePlanningError> {
    if path == StaticSolvePath::ElasticRigidityMultigrid && !elastic_rigidity_admits(input) {
        return Err(ResourcePlanningError::Invalid(
            "the elastic rigidity path requires a linear static tetrahedral solid with fixed-DOF constraints".into(),
        ));
    }
    Ok(())
}

fn build_static_resource_input_with_facts(
    input: StaticResourcePreflight<'_>,
    path: StaticSolvePath,
    compiler_peak: ByteCount,
    facts: &StructuralResourceFacts,
) -> Result<FeaResourcePlanningInput, ResourcePlanningError> {
    validate_static_path(&input, path)?;
    let StaticResourcePreflight {
        compilation: _,
        model,
        step,
        backend,
        preconditioner,
        observation_execution,
        mesh,
        constraints,
        observations,
        accelerator_requests,
        local_execution,
        prepared_inputs,
    } = input;
    let state_profiles_output =
        super::structural_state_profiles_output_resource_facts(model, step, facts.topology_digest)?;
    let thermal_storage = super::thermal_state::ThermalStateStorage::derive(step, mesh)?;
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
        facts.connector_assembly_buffer_bytes,
    )?;
    let observations = ObservationResourceFacts::from_plan(observations, observation_execution)?;
    let recovery_topology = StructuralRecoveredFieldTopology::derive(model, mesh)
        .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?;
    let field_recovery = FieldRecoveryResourceFacts::from_plan(
        &FieldRecoveryPlan::resolve_structural(step, recovery_topology, constraints.groups.len())
            .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?,
    )?;
    let dofs = facts.sparse.rows.0;
    let stiffness_nonzeros = facts.sparse.maximum_nonzeros();
    // Newton always solves Tᵀ K T, including direct fixed-DOF constraints;
    // linear direct elimination instead retains the full matrix dimension.
    let admission_pattern = if constraints.enforcement != ConstraintEnforcement::DirectElimination
        || step.procedure == runmat_analysis_core::AnalysisProcedure::Nonlinear
    {
        std::borrow::Cow::Owned(
            facts
                .constraint_matrix
                .affine_reduction()
                .and_then(|reduction| {
                    reduction.reduce_sparse_pattern(facts.symbolic.stiffness_upper_bound())
                })
                .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?,
        )
    } else {
        std::borrow::Cow::Borrowed(facts.symbolic.stiffness_upper_bound())
    };
    // The topology-derived pattern is a superset of the prepared operator's graph, but the
    // fill-reducing ordering is recalculated for the exact graph. Even a successful symbolic
    // proof on the superset is therefore not an upper bound on exact fill or update work.
    // Reserve the graph-count envelope; exact coercivity is proved after numerical assembly.
    // The planned path decides the proof: the proving factorization reserves the factor; the
    // elastic rigidity path reserves only its element and anchor inventory.
    let admission = match (path, mesh) {
        (StaticSolvePath::ElasticRigidityMultigrid, Some(mesh)) => {
            elastic_rigidity_admission(mesh, admission_pattern.rows() as u64)
        }
        _ => factor_admission(&admission_pattern)?,
    };
    let solve = solve_policy(
        preconditioner,
        path,
        dofs,
        u64::try_from(facts.constraint_matrix.row_count())
            .map_err(|_| ResourcePlanningError::Overflow("constraint row count"))?,
        constraints.enforcement,
        stiffness_nonzeros,
        admission,
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
    let publishes_linearization = linearization::requires_envelope(step);
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
    recovery_values = checked_add(
        recovery_values,
        thermal_storage.output_values,
        "solved thermal recovery values",
    )?;
    if publishes_linearization {
        let linearization_values =
            linearization::value_envelope(dofs, facts.topology.elements.0, stiffness_nonzeros)?;
        recovery_values = recovery_values.checked_add(linearization_values).ok_or(
            ResourcePlanningError::Overflow("structural result value count"),
        )?;
    }
    recovery_values = recovery_values
        .checked_add(facts.connector_recovery_value_count)
        .ok_or(ResourcePlanningError::Overflow(
            "connector recovery value count",
        ))?;
    let recovery_bytes = checked_bytes(recovery_values, 8, "structural recovery bytes")?;
    let recovery_payload_bytes = checked_add(
        recovery_bytes,
        facts.connector_recovery_retained_bytes,
        "structural and connector recovery payload bytes",
    )?;
    let recovery_resident = checked_add(
        recovery_payload_bytes,
        STRUCTURAL_ARTIFACT_OVERHEAD_BYTES,
        "structural recovery resident bytes",
    )?;
    let checkpoint_bytes = checked_add(
        checked_bytes(dofs, 8, "structural checkpoint values")?,
        STRUCTURAL_ARTIFACT_OVERHEAD_BYTES,
        "structural checkpoint bytes",
    )?;
    let accelerators = structural_accelerator_requirements(backend, accelerator_requests)?;
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
        .checked_fixed_scratch_bytes(ByteCount(FIXED_ASSEMBLY_SCRATCH_BYTES_PER_WORKER))
        .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?;
    Ok(FeaResourcePlanningInput {
        estimation_model_revision: RESOURCE_ESTIMATION_MODEL_REVISION,
        phase: FeaResourcePhase::StructuralLinearPipeline,
        demand: FeaResourceDemand::StructuralLinearPipeline(
            StructuralLinearPipelineResourceDemand {
                constraint_compilation_peak_bytes: compiler_peak,
                inputs: prepared_inputs.cloned(),
                state_profiles_output,
                operator: StructuralOperatorResourceDemand {
                    excitation_workspace: super::excitation_workspace::derive_excitation_workspace(
                        model, step, mesh,
                    )?,
                    topology_digest: facts.topology_digest,
                    numerical_scope_digest: facts.symbolic.identity(),
                    topology: facts.topology.clone(),
                    sparse: facts.sparse.clone(),
                    constraints: facts.constraints.clone(),
                    assembly: AssemblyResourcePolicy {
                        contribution_nonzero_upper_bound: Count(contribution_nonzeros),
                        contribution_bytes_upper_bound: ByteCount(contribution_bytes),
                        scratch_bytes_per_element: ByteCount(ASSEMBLY_SCRATCH_BYTES_PER_ELEMENT),
                        fixed_scratch_bytes_per_worker: fixed_assembly_scratch,
                    },
                    solve,
                    phase_resources: PhaseResourcePolicy {
                        resident_value_count: Count(recovery_values),
                        resident_bytes_upper_bound: ByteCount(recovery_resident),
                        scratch_bytes: ByteCount(recovery_resident),
                        retained_bytes: ByteCount(recovery_payload_bytes),
                    },
                    field_recovery,
                    observations: observations.clone(),
                    publication: CheckpointArtifactPolicy {
                        result_artifact_bytes: ByteCount(recovery_payload_bytes),
                        checkpoint_bytes: ByteCount(checkpoint_bytes),
                        artifact_overhead_bytes: ByteCount(STRUCTURAL_ARTIFACT_OVERHEAD_BYTES),
                        relay_bytes: ByteCount(0),
                        result_egress_bytes: ByteCount(recovery_payload_bytes),
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
                },
            },
        ),
        output_set: None,
        worker: WorkerResourcePolicy {
            cpu_millicores,
            maximum_wall_milliseconds: step.accuracy.resources.maximum_wall_time_milliseconds,
            fixed_memory_bytes: ByteCount(checked_add(
                STRUCTURAL_FIXED_WORKER_MEMORY_BYTES,
                thermal_storage.resident_bytes,
                "worker and solved thermal numerical storage",
            )?),
            fixed_scratch_bytes: ByteCount(STRUCTURAL_FIXED_WORKER_SCRATCH_BYTES),
            required_capabilities,
            accelerators,
        },
        accuracy_ceiling: step.accuracy.resources,
    })
}

fn structural_accelerator_requirements(
    backend: crate::ComputeBackend,
    requests: &[AcceleratorRequest],
) -> Result<Vec<super::AcceleratorRequirement>, ResourcePlanningError> {
    if backend != crate::ComputeBackend::Gpu {
        return Ok(Vec::new());
    }
    let [request] = requests else {
        return Err(ResourcePlanningError::Invalid(
            "GPU structural execution requires exactly one accelerator request".into(),
        ));
    };
    if request.count != 1 || request.minimum_allocation_bytes == 0 {
        return Err(ResourcePlanningError::Invalid(
            "the single-device structural envelope requires one accelerator with bounded memory"
                .into(),
        ));
    }
    Ok(vec![super::AcceleratorRequirement {
        class: request.class.clone(),
        count: request.count,
        memory_bytes_each: ByteCount(request.minimum_allocation_bytes),
    }])
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
