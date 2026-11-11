use std::collections::BTreeSet;

use runmat_analysis_fea::constraints::CanonicalStructuralConstraintPlan;
use runmat_analysis_fea::execution::{
    AnalysisNumericalObjectKind, AnalysisWorkKind, CanonicalSemanticState, ConstraintPlanIdentity,
    DiscretizationIdentity, ExecutionPlanIdentity, NumericalPlanIdentity, OperatorIdentity,
    ProblemIdentity, SequenceIdentity, SignalMaterializationIdentity, SignalRealizationIdentity,
    SolverExcitationIdentity, SolverResultIdentity, StepIdentity,
    StructuralLinearNumericalReceiptArtifact, StructuralLinearStateInput, StructuralLinearWorkload,
    StructuralLinearWorkloadResult, StudyIdentity,
};
use runmat_analysis_fea::planning::resource::{
    build_structural_resource_plan_from_compilation, require_executable_static_backend,
    FeaResourcePlan, LocalExecutionPlan,
};
use runmat_analysis_fea::post::fields::structural_static_observable_field_definitions;
use runmat_analysis_fea::results::observation::{
    resolve_structural_observation_plan, ObservationLimits,
};
use runmat_canonical_codec::{encode_contract, CanonicalLimits};
use runmat_execution::resource::AcceleratorRequest;
use runmat_execution::value::ValueRefKind;
use runmat_execution::{Digest, ProgramRevision};
use runmat_execution_artifact::cache::CacheImport;
use runmat_execution_artifact::object::ObjectInventoryLimits;
use runmat_execution_artifact::{ArtifactAccess, ObjectNamespace, ProgramExecutionResponse};
use runmat_meshing_core::{CanonicalMeshingContract, SolverMeshArtifact};
use serde::Serialize;

use crate::{
    bind_upstream_state_transition, import_analysis_objects, import_analysis_output_set,
    prepare_analysis_objects, prepare_analysis_program, AnalysisBridgeError, AnalysisBridgeResult,
    AnalysisCapabilityRequirement, AnalysisHostWorkload, AnalysisWorkloadInput,
    AnalysisWorkloadRequest, PreparedAnalysisObjects, PreparedAnalysisProgram, ANALYSIS_HOST_ABI,
};

const PLAN_PREFIX: &[u8] = b"runmat-analysis-structural-plan/v1\0";
const PLAN_LIMITS: CanonicalLimits = CanonicalLimits::new(64 * 1024 * 1024, 1_000_000, 65_536, 64);

pub struct StructuralLinearProgramPreparation {
    pub authored_study_digest: Digest,
    pub frozen_project_digest: Option<Digest>,
    pub program_revision: ProgramRevision,
    pub artifact_access: ArtifactAccess,
    pub workload: StructuralLinearWorkload,
    pub mesh: Option<SolverMeshArtifact>,
    pub state_input: Option<PreparedAnalysisObjects>,
    pub chunk_bytes: usize,
    pub inventory_limits: ObjectInventoryLimits,
    pub required_capabilities: BTreeSet<AnalysisCapabilityRequirement>,
    pub accelerator_requests: Vec<AcceleratorRequest>,
    pub local_execution: LocalExecutionPlan,
    /// How the static solve path is chosen: within the effective ceiling for a fresh solve, or
    /// the path an admitted plan recorded when a completion is imported or re-prepared.
    pub solve_selection: runmat_analysis_fea::planning::resource::StaticSolveSelection,
}

pub struct PreparedStructuralLinearProgram {
    pub host: AnalysisHostWorkload,
    pub program: PreparedAnalysisProgram,
    pub inputs: Vec<PreparedAnalysisObjects>,
    pub resource_plan: FeaResourcePlan,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ImportedStructuralLinearResult {
    pub state_publication_profiles: runmat_analysis_fea::execution::StatePublicationProfiles,
    pub state_publication_profile_objects: PreparedAnalysisObjects,
    pub identity: SolverResultIdentity,
    pub result: StructuralLinearWorkloadResult,
    pub objects: PreparedAnalysisObjects,
    pub numerical_receipt: StructuralLinearNumericalReceiptArtifact,
    pub numerical_receipt_objects: PreparedAnalysisObjects,
    pub result_object_bytes: u64,
}

/// Resolves one authored sequence step against an optional completed upstream state before
/// preparing its immutable solver task. Scheduling and placement cannot enter this projection.
pub fn prepare_structural_sequence_step(
    mut preparation: StructuralLinearProgramPreparation,
) -> AnalysisBridgeResult<PreparedStructuralLinearProgram> {
    if preparation.workload.state_input.is_some() {
        return Err(AnalysisBridgeError::Invalid(
            "sequence-step preparation derives its transition and does not accept a prebound one"
                .into(),
        ));
    }
    let step = preparation.workload.procedure_step()?.clone();
    match step.state_inputs.as_slice() {
        [] => {
            if preparation.state_input.is_some() {
                return Err(AnalysisBridgeError::Invalid(
                    "root sequence step cannot receive an upstream semantic state".into(),
                ));
            }
        }
        [input]
            if matches!(
                input.use_as,
                runmat_analysis_core::StateUse::InitialGuess
                    | runmat_analysis_core::StateUse::Prestress
            ) =>
        {
            let state_objects = preparation.state_input.as_ref().ok_or_else(|| {
                AnalysisBridgeError::Invalid(
                    "stateful sequence step is missing its upstream semantic state".into(),
                )
            })?;
            if state_objects.manifest.kind != AnalysisNumericalObjectKind::SemanticState {
                return Err(AnalysisBridgeError::Invalid(
                    "stateful sequence step received a non-state numerical object".into(),
                ));
            }
            let state = CanonicalSemanticState::from_canonical_bytes(&state_objects.payload())?;
            let transition = bind_upstream_state_transition(
                &preparation.workload.model,
                &step.step_id,
                &state,
                input.use_as,
                structural_topology_digest(&preparation.workload, preparation.mesh.as_ref())?,
            )?;
            preparation.workload.state_input = Some(match input.use_as {
                runmat_analysis_core::StateUse::InitialGuess => {
                    StructuralLinearStateInput::InitialGuess(transition)
                }
                runmat_analysis_core::StateUse::Prestress => {
                    StructuralLinearStateInput::ThermalPrestress(transition)
                }
                _ => unreachable!("matched executable structural state input"),
            });
        }
        _ => {
            return Err(AnalysisBridgeError::Invalid(
                "structural sequence execution supports one initial-guess or thermal-prestress state input"
                    .into(),
            ));
        }
    }
    prepare_structural_linear_program(preparation)
}

pub fn prepare_structural_linear_program(
    preparation: StructuralLinearProgramPreparation,
) -> AnalysisBridgeResult<PreparedStructuralLinearProgram> {
    preparation.workload.validate()?;
    require_executable_static_backend(preparation.workload.backend)?;
    if let Some(mesh) = &preparation.mesh {
        mesh.validate()
            .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    }
    if preparation.chunk_bytes == 0 {
        return Err(AnalysisBridgeError::Invalid(
            "structural analysis chunk size must be non-zero".into(),
        ));
    }
    let mesh_revision = u64::from(preparation.workload.model.geometry_revision);
    if preparation
        .mesh
        .as_ref()
        .is_some_and(|mesh| mesh.geometry.geometry_revision != mesh_revision)
    {
        return Err(AnalysisBridgeError::Invalid(
            "structural workload and solver mesh geometry revisions differ".into(),
        ));
    }

    let compilation = CanonicalStructuralConstraintPlan::compile_for_model_with_workspace_limit(
        &preparation.workload.model,
        preparation.mesh.as_ref(),
        preparation
            .workload
            .procedure_step()?
            .accuracy
            .resources
            .resident_byte_budget(),
    )
    .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let constraint_plan = compilation.plan();
    let property_map = runmat_analysis_fea::properties::compile_property_map(
        &preparation.workload.model,
        preparation.mesh.as_ref(),
    )
    .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let constraint_plan_bytes = constraint_plan
        .canonical_bytes()
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let constraint_plan_digest = constraint_plan
        .content_digest()
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let field_definitions = structural_static_observable_field_definitions()
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let observation_plan = resolve_structural_observation_plan(
        preparation.workload.procedure_step()?,
        &preparation.workload.model,
        preparation.mesh.as_ref(),
        constraint_plan,
        &field_definitions,
        ObservationLimits::default(),
    )
    .map_err(AnalysisBridgeError::from)?;
    let solve_bytes = preparation.workload.canonical_bytes()?;
    let topology_digest = constraint_plan.topology.discretization_digest();
    match (
        preparation
            .workload
            .state_input
            .as_ref()
            .map(StructuralLinearStateInput::transition),
        preparation.state_input.as_ref(),
    ) {
        (Some(transition), Some(state))
            if state.manifest.kind == AnalysisNumericalObjectKind::SemanticState
                && state.manifest.logical_identity == transition.source_state.digest() =>
        {
            let decoded =
                runmat_analysis_fea::execution::CanonicalSemanticState::from_canonical_bytes(
                    &state.payload(),
                )?;
            if decoded.identity != transition.source_state
                || decoded.topology_digest != topology_digest
                || transition.target_topology_digest != topology_digest
            {
                return Err(AnalysisBridgeError::Identity(
                    "structural source state differs from its transition or solver mesh",
                ));
            }
            let use_as = match preparation.workload.state_input.as_ref() {
                Some(StructuralLinearStateInput::InitialGuess(_)) => {
                    runmat_analysis_core::StateUse::InitialGuess
                }
                Some(StructuralLinearStateInput::ThermalPrestress(_)) => {
                    runmat_analysis_core::StateUse::Prestress
                }
                None => {
                    return Err(AnalysisBridgeError::Identity(
                        "structural source has no bound input",
                    ))
                }
            };
            let rebound = bind_upstream_state_transition(
                &preparation.workload.model,
                &preparation.workload.step_id,
                &decoded,
                use_as,
                topology_digest,
            )?;
            if &rebound != transition {
                return Err(AnalysisBridgeError::Identity(
                    "structural source does not reproduce its bound transition",
                ));
            }
        }
        (None, None) => {}
        _ => {
            return Err(AnalysisBridgeError::Invalid(
                "structural transition and source state must be supplied together".into(),
            ));
        }
    }
    let identities = derive_identities(
        &preparation.workload,
        preparation.authored_study_digest,
        preparation.frozen_project_digest,
        topology_digest,
        constraint_plan_digest,
        property_map.identity.semantic_digest,
    )?;
    if let Some(transition) = preparation
        .workload
        .state_input
        .as_ref()
        .map(StructuralLinearStateInput::transition)
    {
        let producer = preparation
            .workload
            .model
            .sequence
            .steps
            .iter()
            .find(|step| step.step_id == transition.producer_step_id)
            .ok_or_else(|| {
                AnalysisBridgeError::Invalid(
                    "structural state producer is absent from the authored sequence".into(),
                )
            })?;
        let expected_producer = StepIdentity::from_solver_contract(identities.sequence, producer)?;
        if transition.sequence != identities.sequence
            || transition.producer != expected_producer
            || transition.consumer != identities.step
        {
            return Err(AnalysisBridgeError::Identity(
                "structural transition identities differ from the authored sequence",
            ));
        }
    }
    let invocation = prepare_analysis_objects(
        AnalysisNumericalObjectKind::SolverInvocation,
        identities.problem.digest(),
        &solve_bytes,
        preparation.chunk_bytes,
        ObjectNamespace::InputValue,
        preparation.inventory_limits,
    )?;
    let constraint_objects = prepare_analysis_objects(
        AnalysisNumericalObjectKind::ConstraintPlan,
        identities.constraint_plan.digest(),
        &constraint_plan_bytes,
        preparation.chunk_bytes,
        ObjectNamespace::InputValue,
        preparation.inventory_limits,
    )?;
    let mut inputs = vec![(invocation, ValueRefKind::DriverObject)];
    if let Some(mesh) = preparation.mesh.as_ref() {
        let mesh_bytes = mesh
            .canonical_encode()
            .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
        inputs.push((
            prepare_analysis_objects(
                AnalysisNumericalObjectKind::Mesh,
                Digest::from_bytes(*mesh.canonical_digest.bytes()),
                &mesh_bytes,
                preparation.chunk_bytes,
                ObjectNamespace::InputValue,
                preparation.inventory_limits,
            )?,
            ValueRefKind::DriverObject,
        ));
    }
    inputs.push((constraint_objects, ValueRefKind::DriverObject));
    if let Some(state_input) = preparation.state_input {
        inputs.push((state_input, ValueRefKind::ResultObject));
    }
    let input_facts = crate::object::prepared_input_resource_facts_from_refs(
        inputs.iter().map(|(objects, _)| objects),
    )?;
    let resource_plan = build_structural_resource_plan_from_compilation(
        preparation.mesh.as_ref(),
        &compilation,
        &observation_plan,
        &preparation.workload,
        &preparation.accelerator_requests,
        preparation.local_execution,
        Some(&input_facts),
        preparation.solve_selection,
    )?;
    let resource_plan_objects = prepare_analysis_objects(
        AnalysisNumericalObjectKind::ResourcePlan,
        resource_plan.identity.digest(),
        &resource_plan.canonical_bytes()?,
        preparation.chunk_bytes,
        ObjectNamespace::InputValue,
        preparation.inventory_limits,
    )?;
    let plan_index = inputs
        .iter()
        .position(|(input, _)| input.manifest.kind == AnalysisNumericalObjectKind::SemanticState)
        .unwrap_or(inputs.len());
    inputs.insert(
        plan_index,
        (resource_plan_objects, ValueRefKind::DriverObject),
    );
    let execution_plan = ExecutionPlanIdentity::new(
        identities.numerical,
        contract_digest(
            "execution",
            &(
                ANALYSIS_HOST_ABI,
                preparation.workload.require_solver_mesh_for_solid,
                AnalysisNumericalObjectKind::SolverResult,
                resource_plan.identity.digest(),
            ),
        )?,
    )?;
    let workload_inputs = inputs
        .iter()
        .map(|(input, reference_kind)| AnalysisWorkloadInput {
            kind: input.manifest.kind,
            logical_identity: input.manifest.logical_identity,
            payload_digest: input.manifest.payload_digest,
            manifest_digest: input.root.digest,
            manifest_encoded_length: input.root.encoded_length,
            reference_kind: *reference_kind,
        })
        .collect();
    let inputs = inputs
        .into_iter()
        .map(|(input, _)| input)
        .collect::<Vec<_>>();
    let mut required_capabilities = preparation.required_capabilities;
    for accelerator in &resource_plan.request.accelerators {
        required_capabilities.insert(AnalysisCapabilityRequirement::Accelerator {
            class: accelerator.class.clone(),
        });
    }
    required_capabilities.insert(AnalysisCapabilityRequirement::Host {
        abi: ANALYSIS_HOST_ABI.into(),
    });
    let workload = AnalysisWorkloadRequest::new(
        identities.study,
        identities.step,
        execution_plan,
        AnalysisWorkKind::Solve,
        workload_inputs,
        vec![
            AnalysisNumericalObjectKind::SolverResult,
            AnalysisNumericalObjectKind::NumericalReceipt,
            AnalysisNumericalObjectKind::StatePublicationProfiles,
        ],
        required_capabilities,
    )?;
    let host = AnalysisHostWorkload::new(
        workload,
        preparation.artifact_access,
        crate::AnalysisOutputPolicy::new(preparation.chunk_bytes, preparation.inventory_limits)?,
    )?;
    let program = prepare_analysis_program(&host, preparation.program_revision, &inputs)?;
    Ok(PreparedStructuralLinearProgram {
        host,
        program,
        inputs,
        resource_plan,
    })
}

mod import;
pub(crate) mod result_source;
pub use import::{import_structural_linear_completion, import_structural_linear_result};

struct StructuralIdentities {
    study: StudyIdentity,
    sequence: SequenceIdentity,
    step: StepIdentity,
    problem: ProblemIdentity,
    constraint_plan: ConstraintPlanIdentity,
    numerical: NumericalPlanIdentity,
}

fn structural_topology_digest(
    workload: &StructuralLinearWorkload,
    mesh: Option<&SolverMeshArtifact>,
) -> AnalysisBridgeResult<Digest> {
    CanonicalStructuralConstraintPlan::compile_for_model_with_workspace_limit(
        &workload.model,
        mesh,
        workload
            .procedure_step()?
            .accuracy
            .resources
            .resident_byte_budget(),
    )
    .map(|compiled| compiled.plan().topology.discretization_digest())
    .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))
}

fn derive_identities(
    workload: &StructuralLinearWorkload,
    authored_study_digest: Digest,
    frozen_project_digest: Option<Digest>,
    topology_digest: Digest,
    constraint_plan_digest: Digest,
    property_semantic_digest: Digest,
) -> AnalysisBridgeResult<StructuralIdentities> {
    let step_contract = workload.procedure_step()?;
    let study = StudyIdentity::from_authored_digest(authored_study_digest, frozen_project_digest)?;
    let sequence = SequenceIdentity::from_solver_contract(&workload.model.sequence)?;
    let step = StepIdentity::from_solver_contract(sequence, step_contract)?;
    let constraints = ConstraintPlanIdentity::new(step, topology_digest, constraint_plan_digest)?;
    let excitations = workload
        .model
        .excitations
        .iter()
        .map(|excitation| {
            let authored = contract_digest("excitation", excitation)?;
            let materialization = SignalMaterializationIdentity::new(
                authored,
                None,
                contract_digest("static-sampling", &step_contract.independent_domain)?,
            )?;
            let signal = SignalRealizationIdentity::new(materialization, authored)?;
            SolverExcitationIdentity::new(
                step,
                signal,
                contract_digest("excitation-target", &excitation.excitation().target)?,
                contract_digest(
                    "canonical-structural-projection",
                    &(topology_digest, authored),
                )?,
            )
            .map_err(Into::into)
        })
        .collect::<AnalysisBridgeResult<Vec<_>>>()?;
    let problem = ProblemIdentity::new(
        step,
        contract_digest("solver-model", &workload.model.solver_identity_contract())?,
        workload
            .state_input
            .as_ref()
            .map(|input| input.transition().identity),
        constraints,
        excitations,
    )?;
    let discretization = DiscretizationIdentity::new(
        problem,
        topology_digest,
        contract_digest("formulation", &topology_digest)?,
    )?;
    let operator = OperatorIdentity::new(
        discretization,
        contract_digest(
            "operator",
            &(property_semantic_digest, &workload.model.boundaries),
        )?,
    )?;
    let numerical = NumericalPlanIdentity::new(
        operator,
        contract_digest(
            "solver",
            &(
                workload.backend,
                workload.preconditioner,
                workload.algebra_backend,
            ),
        )?,
    )?;
    Ok(StructuralIdentities {
        study,
        sequence,
        step,
        problem,
        constraint_plan: constraints,
        numerical,
    })
}

fn contract_digest<T: Serialize>(domain: &'static str, value: &T) -> AnalysisBridgeResult<Digest> {
    Ok(Digest::sha256(encode_contract(
        PLAN_PREFIX,
        domain,
        value,
        PLAN_LIMITS,
    )?))
}

fn validate_result_semantics(
    invocation: &PreparedAnalysisObjects,
    constraint_plan: &PreparedAnalysisObjects,
    result: &StructuralLinearWorkloadResult,
) -> AnalysisBridgeResult<()> {
    let workload = StructuralLinearWorkload::from_canonical_bytes(&invocation.payload())?;
    let constraint_plan =
        CanonicalStructuralConstraintPlan::from_canonical_bytes(&constraint_plan.payload())
            .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    result
        .constraint_result
        .validate_against(&constraint_plan)
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let connector_result =
        result
            .run
            .structural_connector_result
            .as_deref()
            .ok_or(AnalysisBridgeError::Identity(
                "structural result has no canonical connector result",
            ))?;
    if connector_result.plan.topology != constraint_plan.topology {
        return Err(AnalysisBridgeError::Identity(
            "structural connector result topology differs from the prepared workload",
        ));
    }
    connector_result
        .plan
        .validate_against_authored(&workload.model.connectors)
        .map_err(|_| {
            AnalysisBridgeError::Identity(
                "structural connector result differs from the authored connector contract",
            )
        })?;
    let step = workload.procedure_step()?;
    result.validate_thermal_workload(&workload)?;
    if result.run.requested_backend != workload.backend
        || result.run.identity.family() != step.family
        || result.run.identity.procedure() != step.procedure
        || result.run.identity.solution_space() != step.solution_space
    {
        return Err(AnalysisBridgeError::Identity(
            "structural result semantics differ from the prepared workload",
        ));
    }
    Ok(())
}
