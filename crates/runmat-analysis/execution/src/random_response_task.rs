use runmat_analysis_fea::{
    constraints::CanonicalStructuralConstraintPlan,
    execution::{
        AnalysisNumericalObjectKind, AnalysisWorkKind, CanonicalSemanticState,
        ConstraintPlanIdentity, DiscretizationIdentity, ExecutionPlanIdentity,
        NumericalPlanIdentity, OperatorIdentity, ProblemIdentity, SequenceIdentity,
        SolverResultIdentity, StepIdentity, StructuralRandomResponseAssemblyRequest,
        StructuralRandomResponseIntervalRequest, StructuralRandomResponseIntervalResult,
        StructuralRandomResponseNumericalReceipt, StructuralRandomResponseSolverResult,
        StructuralRandomResponseWorkload, StudyIdentity,
    },
    planning::resource::{
        structural_random_response_assembly_payload_bytes_upper_bound,
        structural_random_response_interval_payload_bytes_upper_bound,
        structural_random_response_receipt_payload_bytes_upper_bound, FeaResourcePlan,
    },
};
use runmat_canonical_codec::{encode_contract, CanonicalLimits};
use runmat_execution::{value::ValueRefKind, Digest, ProgramRevision};
use runmat_execution_artifact::{
    cache::CacheImport, object::ObjectInventoryLimits, ArtifactAccess, ObjectNamespace,
    ProgramExecutionResponse,
};
use runmat_meshing_core::{CanonicalMeshingContract, SolverMeshArtifact};
use serde::Serialize;

use crate::portable_task_support::{resource_plan_objects, task_host};
use crate::{
    decode_prepared_structural_random_response_interval_result,
    decode_prepared_structural_random_response_numerical_receipt,
    decode_prepared_structural_random_response_solver_result, import_analysis_objects,
    import_analysis_output_set, prepare_analysis_objects, prepare_analysis_program,
    prepare_semantic_state_objects, prepare_structural_random_response_assembly_request_objects,
    prepare_structural_random_response_interval_request_objects,
    prepare_structural_random_response_workload_objects, prepared_input_resource_facts,
    projected_output_resource_facts, projected_output_set_resource_facts, AnalysisBridgeError,
    AnalysisBridgeResult, AnalysisHostWorkload, PreparedAnalysisObjects, PreparedAnalysisProgram,
};

const TASK_PREFIX: &[u8] = b"runmat-analysis-structural-random-response-task/v1\0";
const TASK_LIMITS: CanonicalLimits = CanonicalLimits::new(64 * 1024 * 1024, 1_000_000, 65_536, 64);

pub struct StructuralRandomResponseFanoutPreparation {
    pub local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan,
    pub authored_study_digest: Digest,
    pub frozen_project_digest: Option<Digest>,
    pub program_revision: ProgramRevision,
    pub artifact_access: ArtifactAccess,
    pub workload: StructuralRandomResponseWorkload,
    pub mesh: SolverMeshArtifact,
    pub modal_state: CanonicalSemanticState,
    pub chunk_bytes: usize,
    pub inventory_limits: ObjectInventoryLimits,
}

pub struct PreparedStructuralRandomResponseIntervalTask {
    pub workload: StructuralRandomResponseWorkload,
    pub request: StructuralRandomResponseIntervalRequest,
    pub resource_plan: FeaResourcePlan,
    pub host: AnalysisHostWorkload,
    pub program: PreparedAnalysisProgram,
    pub inputs: Vec<PreparedAnalysisObjects>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ImportedStructuralRandomResponseInterval {
    pub result: StructuralRandomResponseIntervalResult,
    pub objects: PreparedAnalysisObjects,
}

pub fn prepare_structural_random_response_fanout(
    preparation: StructuralRandomResponseFanoutPreparation,
) -> AnalysisBridgeResult<Vec<PreparedStructuralRandomResponseIntervalTask>> {
    if preparation.chunk_bytes == 0 {
        return Err(AnalysisBridgeError::Invalid(
            "random-response fan-out requires bounded nonzero chunks".into(),
        ));
    }
    let constraints = CanonicalStructuralConstraintPlan::compile_for_model(
        preparation.workload.model(),
        Some(&preparation.mesh),
    )
    .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let prepared = runmat_analysis_fea::prepare_structural_random_response(
        &preparation.workload,
        &preparation.mesh,
        &preparation.modal_state,
        &constraints,
    )?;
    let topology_digest = Digest::from_bytes(*preparation.mesh.canonical_digest.bytes());
    let sequence = SequenceIdentity::from_solver_contract(&preparation.workload.model().sequence)?;
    let step =
        StepIdentity::from_solver_contract(sequence, preparation.workload.procedure_step()?)?;
    let constraint_identity = ConstraintPlanIdentity::new(
        step,
        topology_digest,
        constraints
            .content_digest()
            .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?,
    )?;
    let workload_objects = prepare_structural_random_response_workload_objects(
        &preparation.workload,
        preparation.chunk_bytes,
        ObjectNamespace::InputValue,
        preparation.inventory_limits,
    )?;
    let mesh_objects = prepare_analysis_objects(
        AnalysisNumericalObjectKind::Mesh,
        topology_digest,
        &preparation
            .mesh
            .canonical_encode()
            .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?,
        preparation.chunk_bytes,
        ObjectNamespace::InputValue,
        preparation.inventory_limits,
    )?;
    let state_objects = prepare_semantic_state_objects(
        &preparation.modal_state,
        preparation.chunk_bytes,
        ObjectNamespace::InputValue,
        preparation.inventory_limits,
    )?;
    let constraint_objects = prepare_analysis_objects(
        AnalysisNumericalObjectKind::ConstraintPlan,
        constraint_identity.digest(),
        &constraints
            .canonical_bytes()
            .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?,
        preparation.chunk_bytes,
        ObjectNamespace::InputValue,
        preparation.inventory_limits,
    )?;
    let study = StudyIdentity::from_authored_digest(
        preparation.authored_study_digest,
        preparation.frozen_project_digest,
    )?;
    let shared = vec![
        workload_objects,
        mesh_objects,
        state_objects,
        constraint_objects,
    ];
    (0..prepared.scope().task_ranges().len())
        .map(|ordinal| {
            let request = StructuralRandomResponseIntervalRequest::new(
                &preparation.workload,
                prepared.scope().clone(),
                ordinal,
            )?;
            let request_objects = prepare_structural_random_response_interval_request_objects(
                &request,
                preparation.chunk_bytes,
                ObjectNamespace::InputValue,
                preparation.inventory_limits,
            )?;
            let mut numerical_inputs = shared.clone();
            numerical_inputs.push(request_objects);
            let resource_plan = prepared
                .resources()
                .plan_for_intervals(
                    &request,
                    prepared_input_resource_facts(&numerical_inputs)?,
                    projected_output_resource_facts(
                        structural_random_response_interval_payload_bytes_upper_bound(&request)
                            .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?,
                    )?,
                    projected_output_set_resource_facts(),
                    preparation.local_execution,
                )
                .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
            let resource_objects = resource_plan_objects(
                &resource_plan,
                preparation.chunk_bytes,
                preparation.inventory_limits,
            )?;
            let execution_plan = execution_plan(
                &preparation.workload,
                prepared.scope(),
                constraint_identity,
                step,
                topology_digest,
                &resource_plan,
            )?;
            numerical_inputs.push(resource_objects);
            let host = task_host(crate::portable_task_support::PortableTaskHostPreparation {
                study,
                step,
                execution_plan,
                work_kind: AnalysisWorkKind::SolveStructuralRandomResponseIntervals,
                inputs: &numerical_inputs,
                outputs: vec![AnalysisNumericalObjectKind::StructuralRandomResponseIntervalResult],
                artifact_access: preparation.artifact_access.clone(),
                output_policy: crate::AnalysisOutputPolicy::new(
                    preparation.chunk_bytes,
                    preparation.inventory_limits,
                )?,
            })?;
            let program = prepare_analysis_program(
                &host,
                preparation.program_revision.clone(),
                &numerical_inputs,
            )?;
            Ok(PreparedStructuralRandomResponseIntervalTask {
                workload: preparation.workload.clone(),
                request,
                resource_plan,
                host,
                program,
                inputs: numerical_inputs,
            })
        })
        .collect()
}

pub fn import_structural_random_response_interval(
    source: &impl CacheImport,
    response: ProgramExecutionResponse,
    prepared: &PreparedStructuralRandomResponseIntervalTask,
    limits: ObjectInventoryLimits,
) -> AnalysisBridgeResult<ImportedStructuralRandomResponseInterval> {
    response.validate_against(&prepared.program.request)?;
    let output_set = import_analysis_output_set(source, response, &prepared.host)?;
    let [output] = output_set.manifest.outputs.as_slice() else {
        return Err(AnalysisBridgeError::Invalid(
            "random-response interval task must return exactly one result".into(),
        ));
    };
    let objects = import_analysis_objects(
        source,
        &output.root,
        AnalysisNumericalObjectKind::StructuralRandomResponseIntervalResult,
        output.logical_identity,
        output.payload_digest,
        &prepared.host.artifact_access,
        limits,
    )?;
    let result = decode_prepared_structural_random_response_interval_result(&objects)?;
    if result.request() != &prepared.request {
        return Err(AnalysisBridgeError::Identity(
            "random-response interval result differs from its prepared request",
        ));
    }
    Ok(ImportedStructuralRandomResponseInterval { result, objects })
}

pub struct StructuralRandomResponseAssemblyPreparation {
    pub local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan,
    pub authored_study_digest: Digest,
    pub frozen_project_digest: Option<Digest>,
    pub program_revision: ProgramRevision,
    pub artifact_access: ArtifactAccess,
    pub workload: StructuralRandomResponseWorkload,
    pub mesh: SolverMeshArtifact,
    pub modal_state: CanonicalSemanticState,
    pub tasks: Vec<PreparedAnalysisObjects>,
    pub chunk_bytes: usize,
    pub inventory_limits: ObjectInventoryLimits,
}

pub struct PreparedStructuralRandomResponseAssemblyTask {
    pub workload: StructuralRandomResponseWorkload,
    pub request: StructuralRandomResponseAssemblyRequest,
    pub resource_plan: FeaResourcePlan,
    pub resource_plan_objects: PreparedAnalysisObjects,
    pub host: AnalysisHostWorkload,
    pub program: PreparedAnalysisProgram,
    pub inputs: Vec<PreparedAnalysisObjects>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ImportedStructuralRandomResponseResult {
    pub state_publication_profiles: runmat_analysis_fea::execution::StatePublicationProfiles,
    pub state_publication_profile_objects: PreparedAnalysisObjects,
    pub identity: SolverResultIdentity,
    pub result: StructuralRandomResponseSolverResult,
    pub result_objects: PreparedAnalysisObjects,
    pub numerical_receipt: StructuralRandomResponseNumericalReceipt,
    pub receipt_objects: PreparedAnalysisObjects,
    pub response: ProgramExecutionResponse,
}

pub fn prepare_structural_random_response_assembly(
    preparation: StructuralRandomResponseAssemblyPreparation,
) -> AnalysisBridgeResult<PreparedStructuralRandomResponseAssemblyTask> {
    if preparation.chunk_bytes == 0 {
        return Err(AnalysisBridgeError::Invalid(
            "random-response fan-in requires bounded nonzero chunks".into(),
        ));
    }
    let tasks = preparation
        .tasks
        .iter()
        .map(decode_prepared_structural_random_response_interval_result)
        .collect::<AnalysisBridgeResult<Vec<_>>>()?;
    let constraints = CanonicalStructuralConstraintPlan::compile_for_model(
        preparation.workload.model(),
        Some(&preparation.mesh),
    )
    .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let prepared = runmat_analysis_fea::prepare_structural_random_response(
        &preparation.workload,
        &preparation.mesh,
        &preparation.modal_state,
        &constraints,
    )?;
    let topology_digest = Digest::from_bytes(*preparation.mesh.canonical_digest.bytes());
    let sequence = SequenceIdentity::from_solver_contract(&preparation.workload.model().sequence)?;
    let step =
        StepIdentity::from_solver_contract(sequence, preparation.workload.procedure_step()?)?;
    let constraint_identity = ConstraintPlanIdentity::new(
        step,
        topology_digest,
        constraints
            .content_digest()
            .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?,
    )?;
    let request = StructuralRandomResponseAssemblyRequest::new(prepared.scope().clone(), tasks)?;
    let workload_objects = prepare_structural_random_response_workload_objects(
        &preparation.workload,
        preparation.chunk_bytes,
        ObjectNamespace::InputValue,
        preparation.inventory_limits,
    )?;
    let request_objects = prepare_structural_random_response_assembly_request_objects(
        &request,
        preparation.chunk_bytes,
        ObjectNamespace::InputValue,
        preparation.inventory_limits,
    )?;
    let mesh_objects = prepare_analysis_objects(
        AnalysisNumericalObjectKind::Mesh,
        topology_digest,
        &preparation
            .mesh
            .canonical_encode()
            .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?,
        preparation.chunk_bytes,
        ObjectNamespace::InputValue,
        preparation.inventory_limits,
    )?;
    let state_objects = prepare_semantic_state_objects(
        &preparation.modal_state,
        preparation.chunk_bytes,
        ObjectNamespace::InputValue,
        preparation.inventory_limits,
    )?;
    let constraint_objects = prepare_analysis_objects(
        AnalysisNumericalObjectKind::ConstraintPlan,
        constraint_identity.digest(),
        &constraints
            .canonical_bytes()
            .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?,
        preparation.chunk_bytes,
        ObjectNamespace::InputValue,
        preparation.inventory_limits,
    )?;
    let mut numerical_inputs = vec![
        workload_objects,
        mesh_objects,
        state_objects,
        constraint_objects,
        request_objects,
    ];
    numerical_inputs.extend(preparation.tasks);
    let resource_plan = prepared
        .resources()
        .plan_for_assembly(
            &request,
            prepared_input_resource_facts(&numerical_inputs)?,
            projected_output_resource_facts(
                structural_random_response_assembly_payload_bytes_upper_bound(&request)
                    .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?,
            )?,
            projected_output_resource_facts(
                structural_random_response_receipt_payload_bytes_upper_bound(),
            )?,
            crate::structural_state_profiles_output_facts(
                preparation.workload.model(), preparation.workload.procedure_step()?,
                &[runmat_analysis_fea::execution::SemanticStatePublicationSource::StructuralRandomStatistics], request.scope().topology_digest(),
            )?,
            projected_output_set_resource_facts(),
            preparation.local_execution,
        )
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let resource_plan_objects = resource_plan_objects(
        &resource_plan,
        preparation.chunk_bytes,
        preparation.inventory_limits,
    )?;
    numerical_inputs.push(resource_plan_objects.clone());
    let execution_plan = execution_plan(
        &preparation.workload,
        prepared.scope(),
        constraint_identity,
        step,
        topology_digest,
        &resource_plan,
    )?;
    let study = StudyIdentity::from_authored_digest(
        preparation.authored_study_digest,
        preparation.frozen_project_digest,
    )?;
    let host = task_host(crate::portable_task_support::PortableTaskHostPreparation {
        study,
        step,
        execution_plan,
        work_kind: AnalysisWorkKind::AssembleStructuralRandomResponseResult,
        inputs: &numerical_inputs,
        outputs: vec![
            AnalysisNumericalObjectKind::SolverResult,
            AnalysisNumericalObjectKind::NumericalReceipt,
            AnalysisNumericalObjectKind::StatePublicationProfiles,
        ],
        artifact_access: preparation.artifact_access,
        output_policy: crate::AnalysisOutputPolicy::new(
            preparation.chunk_bytes,
            preparation.inventory_limits,
        )?,
    })?;
    let program = prepare_analysis_program(&host, preparation.program_revision, &numerical_inputs)?;
    Ok(PreparedStructuralRandomResponseAssemblyTask {
        workload: preparation.workload,
        request,
        resource_plan,
        resource_plan_objects,
        host,
        program,
        inputs: numerical_inputs,
    })
}

pub fn import_structural_random_response_result(
    source: &impl CacheImport,
    response: ProgramExecutionResponse,
    prepared: &PreparedStructuralRandomResponseAssemblyTask,
    limits: ObjectInventoryLimits,
) -> AnalysisBridgeResult<ImportedStructuralRandomResponseResult> {
    response.validate_against(&prepared.program.request)?;
    let retained_response = response.clone();
    let output_set = import_analysis_output_set(source, response, &prepared.host)?;
    let [result_output, receipt_output, profiles_output] = output_set.manifest.outputs.as_slice()
    else {
        return Err(AnalysisBridgeError::Invalid(
            "random-response fan-in must return result, receipt and publication profiles".into(),
        ));
    };
    let identity = SolverResultIdentity::new(
        prepared.host.workload.identity,
        result_output.payload_digest,
    )?;
    if result_output.logical_identity != identity.digest() {
        return Err(AnalysisBridgeError::Identity(
            "random-response result root differs from its workload-bound identity",
        ));
    }
    let result_objects = import_analysis_objects(
        source,
        &result_output.root,
        AnalysisNumericalObjectKind::SolverResult,
        result_output.logical_identity,
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
    let result = decode_prepared_structural_random_response_solver_result(&result_objects)?;
    result.validate_for(&prepared.workload)?;
    if result.assembly() != &prepared.request {
        return Err(AnalysisBridgeError::Identity(
            "random-response result differs from its prepared fan-in",
        ));
    }
    let numerical_receipt =
        decode_prepared_structural_random_response_numerical_receipt(&receipt_objects)?;
    numerical_receipt.validate_against(identity, &result)?;
    if receipt_output.logical_identity != numerical_receipt.identity().digest() {
        return Err(AnalysisBridgeError::Identity(
            "random-response receipt root differs from its canonical identity",
        ));
    }
    let state_publication_profile_objects = import_analysis_objects(
        source,
        &profiles_output.root,
        AnalysisNumericalObjectKind::StatePublicationProfiles,
        profiles_output.logical_identity,
        profiles_output.payload_digest,
        &prepared.host.artifact_access,
        limits,
    )?;
    let state_publication_profiles = crate::structural_state_profiles::validate_profiles(
        &state_publication_profile_objects,
        crate::structural_state_publication_profiles(
            prepared.workload.model(),
            prepared.workload.procedure_step()?,
            identity,
            result_objects.manifest.payload_digest,
            crate::StructuralStateProfileResult::RandomResponse(&result),
        )?,
    )?;
    let mut expected = result_objects
        .unique_references(&prepared.host.artifact_access, ValueRefKind::ResultObject)?;
    expected.extend(
        receipt_objects
            .unique_references(&prepared.host.artifact_access, ValueRefKind::ResultObject)?,
    );
    expected.extend(
        state_publication_profile_objects
            .unique_references(&prepared.host.artifact_access, ValueRefKind::ResultObject)?,
    );
    crate::object::retain_unique_references(&mut expected);
    expected.push(output_set.root);
    if output_set.result_objects != expected {
        return Err(AnalysisBridgeError::Identity(
            "random-response result inventory differs from its canonical closure",
        ));
    }
    Ok(ImportedStructuralRandomResponseResult {
        state_publication_profiles,
        state_publication_profile_objects,
        identity,
        result,
        result_objects,
        numerical_receipt,
        receipt_objects,
        response: retained_response,
    })
}

fn execution_plan(
    workload: &StructuralRandomResponseWorkload,
    scope: &runmat_analysis_fea::execution::StructuralRandomResponseScope,
    constraint_identity: ConstraintPlanIdentity,
    step: StepIdentity,
    topology_digest: Digest,
    resources: &FeaResourcePlan,
) -> AnalysisBridgeResult<ExecutionPlanIdentity> {
    let problem = ProblemIdentity::new(
        step,
        contract_digest(
            "random-response-model",
            &workload.model().solver_identity_contract(),
        )?,
        Some(workload.modal_basis_transition().identity),
        constraint_identity,
        Vec::new(),
    )?;
    let discretization = DiscretizationIdentity::new(
        problem,
        topology_digest,
        contract_digest("random-response-formulation", &topology_digest)?,
    )?;
    let operator = OperatorIdentity::new(discretization, scope.operator_digest())?;
    let numerical = NumericalPlanIdentity::new(
        operator,
        contract_digest(
            "random-response-solver",
            &(
                scope.basis(),
                scope.plan().excitation_digest(),
                scope.plan().policy(),
                scope.plan().breakpoints_hz(),
                workload.backend(),
            ),
        )?,
    )?;
    ExecutionPlanIdentity::new(
        numerical,
        contract_digest("random-response-execution", &resources.identity)?,
    )
    .map_err(Into::into)
}

fn contract_digest<T: Serialize>(domain: &'static str, value: &T) -> AnalysisBridgeResult<Digest> {
    Ok(Digest::sha256(encode_contract(
        TASK_PREFIX,
        domain,
        value,
        TASK_LIMITS,
    )?))
}
