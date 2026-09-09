use std::collections::BTreeSet;

use runmat_analysis_fea::constraints::CanonicalStructuralConstraintPlan;
use runmat_analysis_fea::execution::{
    AnalysisNumericalObjectKind, AnalysisWorkKind, CanonicalSemanticState,
    CanonicalStateTransition, CanonicalStateTransitionInput, ConstraintPlanIdentity,
    DiscretizationIdentity, ExecutionPlanIdentity, NumericalPlanIdentity, OperatorIdentity,
    ProblemIdentity, SequenceIdentity, SignalMaterializationIdentity, SignalRealizationIdentity,
    SolverExcitationIdentity, SolverResultIdentity, StepIdentity,
    StructuralLinearNumericalReceiptArtifact, StructuralLinearWorkload,
    StructuralLinearWorkloadResult, StudyIdentity,
};
use runmat_analysis_fea::planning::resource::{build_structural_resource_plan, FeaResourcePlan};
use runmat_analysis_fea::post::fields::structural_observable_field_definitions;
use runmat_analysis_fea::results::observation::{resolve_observation_plan, ObservationLimits};
use runmat_analysis_fea::topology::AnalysisProjection;
use runmat_canonical_codec::{encode_contract, CanonicalLimits};
use runmat_execution::value::ValueRefKind;
use runmat_execution::{Digest, ProgramRevision};
use runmat_execution_artifact::cache::CacheImport;
use runmat_execution_artifact::object::ObjectInventoryLimits;
use runmat_execution_artifact::{ArtifactAccess, ObjectNamespace, ProgramExecutionResponse};
use runmat_meshing_core::{CanonicalMeshingContract, SolverMeshArtifact};
use serde::Serialize;

use crate::{
    import_analysis_objects, import_analysis_output_set, prepare_analysis_objects,
    prepare_analysis_program, AnalysisBridgeError, AnalysisBridgeResult,
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
    pub mesh: SolverMeshArtifact,
    pub initial_guess: Option<PreparedAnalysisObjects>,
    pub chunk_bytes: usize,
    pub inventory_limits: ObjectInventoryLimits,
    pub required_capabilities: BTreeSet<AnalysisCapabilityRequirement>,
}

pub struct PreparedStructuralLinearProgram {
    pub host: AnalysisHostWorkload,
    pub program: PreparedAnalysisProgram,
    pub inputs: Vec<PreparedAnalysisObjects>,
    pub resource_plan: FeaResourcePlan,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ImportedStructuralLinearResult {
    pub identity: SolverResultIdentity,
    pub result: StructuralLinearWorkloadResult,
    pub objects: PreparedAnalysisObjects,
    pub numerical_receipt: StructuralLinearNumericalReceiptArtifact,
    pub numerical_receipt_objects: PreparedAnalysisObjects,
}

/// Resolves one authored sequence step against an optional completed upstream state before
/// preparing its immutable solver task. Scheduling and placement cannot enter this projection.
pub fn prepare_structural_sequence_step(
    mut preparation: StructuralLinearProgramPreparation,
) -> AnalysisBridgeResult<PreparedStructuralLinearProgram> {
    if preparation.workload.initial_guess_transition.is_some() {
        return Err(AnalysisBridgeError::Invalid(
            "sequence-step preparation derives its transition and does not accept a prebound one"
                .into(),
        ));
    }
    let step = preparation.workload.procedure_step()?.clone();
    match step.state_inputs.as_slice() {
        [] => {
            if preparation.initial_guess.is_some() {
                return Err(AnalysisBridgeError::Invalid(
                    "root sequence step cannot receive an upstream semantic state".into(),
                ));
            }
        }
        [input] if input.use_as == runmat_analysis_core::StateUse::InitialGuess => {
            let runmat_analysis_core::StateSource::Upstream {
                producer_step,
                state_id,
            } = &input.source
            else {
                return Err(AnalysisBridgeError::Invalid(
                    "structural initial guess must come from an upstream sequence state".into(),
                ));
            };
            let state_objects = preparation.initial_guess.as_ref().ok_or_else(|| {
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
            if state.state_id != *state_id || state.kind != input.required_kind {
                return Err(AnalysisBridgeError::Identity(
                    "upstream semantic state differs from the selected state input",
                ));
            }
            let authored_transition = preparation
                .workload
                .model
                .sequence
                .transitions
                .iter()
                .find(|transition| {
                    transition.producer_step == *producer_step
                        && transition.consumer_step == step.step_id
                        && transition.state_id == *state_id
                })
                .ok_or_else(|| {
                    AnalysisBridgeError::Invalid(
                        "stateful sequence step has no matching authored transition".into(),
                    )
                })?;
            let sequence =
                SequenceIdentity::from_solver_contract(&preparation.workload.model.sequence)?;
            let producer_contract = preparation
                .workload
                .model
                .sequence
                .steps
                .iter()
                .find(|candidate| candidate.step_id == *producer_step)
                .ok_or_else(|| {
                    AnalysisBridgeError::Invalid(
                        "state transition producer is absent from the authored sequence".into(),
                    )
                })?;
            let producer = StepIdentity::from_solver_contract(sequence, producer_contract)?;
            let consumer = StepIdentity::from_solver_contract(sequence, &step)?;
            if state.producer != producer {
                return Err(AnalysisBridgeError::Identity(
                    "upstream semantic state names a different producer step",
                ));
            }
            preparation.workload.initial_guess_transition = Some(CanonicalStateTransition::new(
                CanonicalStateTransitionInput {
                    sequence,
                    producer,
                    consumer,
                    authored: authored_transition,
                    source: &state,
                    target_topology_digest: Digest::from_bytes(
                        *preparation.mesh.canonical_digest.bytes(),
                    ),
                },
            )?);
        }
        _ => {
            return Err(AnalysisBridgeError::Invalid(
                "structural sequence execution currently supports at most one initial-guess state input"
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
    preparation
        .mesh
        .validate()
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    if preparation.chunk_bytes == 0 {
        return Err(AnalysisBridgeError::Invalid(
            "structural analysis chunk size must be non-zero".into(),
        ));
    }
    let mesh_revision = u64::from(preparation.workload.model.geometry_revision);
    if preparation.mesh.geometry.geometry_revision != mesh_revision {
        return Err(AnalysisBridgeError::Invalid(
            "structural workload and solver mesh geometry revisions differ".into(),
        ));
    }

    let constraint_plan =
        CanonicalStructuralConstraintPlan::compile(&preparation.workload.model, &preparation.mesh)
            .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let constraint_plan_bytes = constraint_plan
        .canonical_bytes()
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let constraint_plan_digest = constraint_plan
        .content_digest()
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let projection = AnalysisProjection::derive(&preparation.mesh)
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let field_definitions = structural_observable_field_definitions()
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let observation_plan = resolve_observation_plan(
        preparation.workload.procedure_step()?,
        &projection,
        &field_definitions,
        ObservationLimits::default(),
    )
    .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let resource_plan = build_structural_resource_plan(
        &projection,
        &constraint_plan,
        &observation_plan,
        &preparation.workload,
    )
    .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let resource_plan_bytes = resource_plan
        .canonical_bytes()
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let solve_bytes = preparation.workload.canonical_bytes()?;
    let mesh_bytes = preparation
        .mesh
        .canonical_encode()
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let mesh_digest = Digest::from_bytes(*preparation.mesh.canonical_digest.bytes());
    match (
        preparation.workload.initial_guess_transition.as_ref(),
        preparation.initial_guess.as_ref(),
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
                || decoded.topology_digest != mesh_digest
                || transition.target_topology_digest != mesh_digest
            {
                return Err(AnalysisBridgeError::Identity(
                    "initial-guess state differs from its transition or solver mesh",
                ));
            }
        }
        (None, None) => {}
        _ => {
            return Err(AnalysisBridgeError::Invalid(
                "structural initial-guess transition and state must be supplied together".into(),
            ));
        }
    }
    let identities = derive_identities(
        &preparation.workload,
        preparation.authored_study_digest,
        preparation.frozen_project_digest,
        mesh_digest,
        constraint_plan_digest,
        resource_plan.identity.digest(),
    )?;
    if let Some(transition) = preparation.workload.initial_guess_transition.as_ref() {
        let producer = preparation
            .workload
            .model
            .sequence
            .steps
            .iter()
            .find(|step| step.step_id == transition.producer_step_id)
            .ok_or_else(|| {
                AnalysisBridgeError::Invalid(
                    "initial-guess transition producer is absent from the authored sequence".into(),
                )
            })?;
        let expected_producer = StepIdentity::from_solver_contract(identities.sequence, producer)?;
        if transition.sequence != identities.sequence
            || transition.producer != expected_producer
            || transition.consumer != identities.step
        {
            return Err(AnalysisBridgeError::Identity(
                "initial-guess transition identities differ from the authored sequence",
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
    let mesh = prepare_analysis_objects(
        AnalysisNumericalObjectKind::Mesh,
        mesh_digest,
        &mesh_bytes,
        preparation.chunk_bytes,
        ObjectNamespace::InputValue,
        preparation.inventory_limits,
    )?;
    let constraint_plan = prepare_analysis_objects(
        AnalysisNumericalObjectKind::ConstraintPlan,
        identities.constraint_plan.digest(),
        &constraint_plan_bytes,
        preparation.chunk_bytes,
        ObjectNamespace::InputValue,
        preparation.inventory_limits,
    )?;
    let resource_plan_objects = prepare_analysis_objects(
        AnalysisNumericalObjectKind::ResourcePlan,
        resource_plan.identity.digest(),
        &resource_plan_bytes,
        preparation.chunk_bytes,
        ObjectNamespace::InputValue,
        preparation.inventory_limits,
    )?;
    let mut inputs = vec![
        (invocation, ValueRefKind::DriverObject),
        (mesh, ValueRefKind::DriverObject),
        (constraint_plan, ValueRefKind::DriverObject),
        (resource_plan_objects, ValueRefKind::DriverObject),
    ];
    if let Some(initial_guess) = preparation.initial_guess {
        inputs.push((initial_guess, ValueRefKind::ResultObject));
    }
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
    required_capabilities.insert(AnalysisCapabilityRequirement::Host {
        abi: ANALYSIS_HOST_ABI.into(),
    });
    let workload = AnalysisWorkloadRequest::new(
        identities.study,
        identities.step,
        identities.execution_plan,
        AnalysisWorkKind::Solve,
        workload_inputs,
        vec![
            AnalysisNumericalObjectKind::SolverResult,
            AnalysisNumericalObjectKind::NumericalReceipt,
        ],
        required_capabilities,
    )?;
    let host = AnalysisHostWorkload::new(workload, preparation.artifact_access)?;
    let program = prepare_analysis_program(&host, preparation.program_revision, &inputs)?;
    Ok(PreparedStructuralLinearProgram {
        host,
        program,
        inputs,
        resource_plan,
    })
}

pub fn import_structural_linear_result(
    source: &impl CacheImport,
    response: ProgramExecutionResponse,
    prepared: &PreparedStructuralLinearProgram,
    limits: ObjectInventoryLimits,
) -> AnalysisBridgeResult<StructuralLinearWorkloadResult> {
    Ok(import_structural_linear_completion(source, response, prepared, limits)?.result)
}

pub fn import_structural_linear_completion(
    source: &impl CacheImport,
    response: ProgramExecutionResponse,
    prepared: &PreparedStructuralLinearProgram,
    limits: ObjectInventoryLimits,
) -> AnalysisBridgeResult<ImportedStructuralLinearResult> {
    response.validate_against(&prepared.program.request)?;
    let output_set = import_analysis_output_set(source, response, &prepared.host)?;
    let [result_output, receipt_output] = output_set.manifest.outputs.as_slice() else {
        return Err(AnalysisBridgeError::Invalid(
            "structural analysis must return result and numerical-receipt roots".into(),
        ));
    };
    let expected_result = SolverResultIdentity::new(
        prepared.host.workload.identity,
        result_output.payload_digest,
    )?;
    let imported = import_analysis_objects(
        source,
        &result_output.root,
        AnalysisNumericalObjectKind::SolverResult,
        expected_result.digest(),
        result_output.payload_digest,
        &prepared.host.artifact_access,
        limits,
    )?;
    let receipt_objects = import_analysis_objects(
        source,
        &receipt_output.root,
        AnalysisNumericalObjectKind::NumericalReceipt,
        receipt_output.logical_identity,
        receipt_output.payload_digest,
        &prepared.host.artifact_access,
        limits,
    )?;
    let mut expected =
        imported.references(&prepared.host.artifact_access, ValueRefKind::ResultObject)?;
    expected.extend(
        receipt_objects.references(&prepared.host.artifact_access, ValueRefKind::ResultObject)?,
    );
    expected.push(output_set.root);
    if output_set.result_objects != expected {
        return Err(AnalysisBridgeError::Identity(
            "analysis result inventory differs from its canonical closure",
        ));
    }
    let result = StructuralLinearWorkloadResult::from_canonical_bytes(&imported.payload())?;
    let numerical_receipt =
        StructuralLinearNumericalReceiptArtifact::from_canonical_bytes(&receipt_objects.payload())?;
    if numerical_receipt.identity().digest() != receipt_output.logical_identity {
        return Err(AnalysisBridgeError::Identity(
            "numerical receipt manifest differs from its canonical identity",
        ));
    }
    numerical_receipt.validate_against(expected_result, &result)?;
    let constraint_plan = prepared
        .inputs
        .iter()
        .find(|input| input.manifest.kind == AnalysisNumericalObjectKind::ConstraintPlan)
        .ok_or_else(|| {
            AnalysisBridgeError::Invalid(
                "prepared structural program has no canonical constraint plan".into(),
            )
        })?;
    validate_result_semantics(&prepared.inputs[0], constraint_plan, &result)?;
    Ok(ImportedStructuralLinearResult {
        identity: expected_result,
        result,
        objects: imported,
        numerical_receipt,
        numerical_receipt_objects: receipt_objects,
    })
}

struct StructuralIdentities {
    study: StudyIdentity,
    sequence: SequenceIdentity,
    step: StepIdentity,
    problem: ProblemIdentity,
    constraint_plan: ConstraintPlanIdentity,
    execution_plan: ExecutionPlanIdentity,
}

fn derive_identities(
    workload: &StructuralLinearWorkload,
    authored_study_digest: Digest,
    frozen_project_digest: Option<Digest>,
    mesh_digest: Digest,
    constraint_plan_digest: Digest,
    resource_plan_digest: Digest,
) -> AnalysisBridgeResult<StructuralIdentities> {
    let step_contract = workload.procedure_step()?;
    let study = StudyIdentity::from_authored_digest(authored_study_digest, frozen_project_digest)?;
    let sequence = SequenceIdentity::from_solver_contract(&workload.model.sequence)?;
    let step = StepIdentity::from_solver_contract(sequence, step_contract)?;
    let constraints = ConstraintPlanIdentity::new(step, mesh_digest, constraint_plan_digest)?;
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
                contract_digest("canonical-mesh-projection", &(mesh_digest, authored))?,
            )
            .map_err(Into::into)
        })
        .collect::<AnalysisBridgeResult<Vec<_>>>()?;
    let problem = ProblemIdentity::new(
        step,
        contract_digest("solver-model", &workload.model.solver_identity_contract())?,
        workload
            .initial_guess_transition
            .as_ref()
            .map(|transition| transition.identity),
        constraints,
        excitations,
    )?;
    let discretization = DiscretizationIdentity::new(
        problem,
        mesh_digest,
        contract_digest(
            "formulation",
            &(workload.formulation_requirement, mesh_digest),
        )?,
    )?;
    let operator = OperatorIdentity::new(
        discretization,
        contract_digest(
            "operator",
            &(
                &workload.model.materials,
                &workload.model.material_assignments,
                &workload.model.boundaries,
            ),
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
    let execution_plan = ExecutionPlanIdentity::new(
        numerical,
        contract_digest(
            "execution",
            &(
                ANALYSIS_HOST_ABI,
                workload.require_solver_mesh_for_solid,
                AnalysisNumericalObjectKind::SolverResult,
                resource_plan_digest,
            ),
        )?,
    )?;
    Ok(StructuralIdentities {
        study,
        sequence,
        step,
        problem,
        constraint_plan: constraints,
        execution_plan,
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
    let step = workload.procedure_step()?;
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
