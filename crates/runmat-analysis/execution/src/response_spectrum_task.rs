use runmat_analysis_fea::{
    constraints::CanonicalStructuralConstraintPlan,
    execution::{
        AnalysisNumericalObjectKind, AnalysisWorkKind, CanonicalSemanticState,
        ConstraintPlanIdentity, DiscretizationIdentity, ExecutionPlanIdentity,
        NumericalPlanIdentity, OperatorIdentity, ProblemIdentity, SequenceIdentity,
        SolverResultIdentity, StepIdentity, StructuralModalResponseSpectrumAssemblyRequest,
        StructuralModalResponseSpectrumDirectionRequest,
        StructuralModalResponseSpectrumDirectionResult,
        StructuralModalResponseSpectrumNumericalReceipt,
        StructuralModalResponseSpectrumSolverResult, StructuralModalResponseSpectrumWorkload,
        StudyIdentity,
    },
    planning::resource::{
        structural_modal_response_spectrum_assembly_payload_bytes_upper_bound,
        structural_modal_response_spectrum_direction_payload_bytes_upper_bound,
        structural_modal_response_spectrum_receipt_payload_bytes_upper_bound, FeaResourcePlan,
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
    decode_prepared_structural_modal_response_spectrum_direction_result,
    decode_prepared_structural_modal_response_spectrum_field_recovery,
    decode_prepared_structural_modal_response_spectrum_numerical_receipt,
    decode_prepared_structural_modal_response_spectrum_solver_result, import_analysis_objects,
    import_analysis_output_set, prepare_analysis_objects, prepare_analysis_program,
    prepare_semantic_state_objects,
    prepare_structural_modal_response_spectrum_assembly_request_objects,
    prepare_structural_modal_response_spectrum_direction_request_objects,
    prepare_structural_modal_response_spectrum_workload_objects, prepared_input_resource_facts,
    projected_output_resource_facts, projected_output_set_resource_facts, AnalysisBridgeError,
    AnalysisBridgeResult, AnalysisHostWorkload, PreparedAnalysisObjects, PreparedAnalysisProgram,
};

const TASK_PREFIX: &[u8] = b"runmat-analysis-structural-modal-response-spectrum-task/v1\0";
const TASK_LIMITS: CanonicalLimits = CanonicalLimits::new(64 * 1024 * 1024, 1_000_000, 65_536, 64);

pub struct StructuralModalResponseSpectrumFanoutPreparation {
    pub authored_study_digest: Digest,
    pub frozen_project_digest: Option<Digest>,
    pub program_revision: ProgramRevision,
    pub artifact_access: ArtifactAccess,
    pub workload: StructuralModalResponseSpectrumWorkload,
    pub mesh: SolverMeshArtifact,
    pub modal_state: CanonicalSemanticState,
    pub chunk_bytes: usize,
    pub inventory_limits: ObjectInventoryLimits,
}

pub struct PreparedStructuralModalResponseSpectrumDirectionTask {
    pub workload: StructuralModalResponseSpectrumWorkload,
    pub request: StructuralModalResponseSpectrumDirectionRequest,
    pub resource_plan: FeaResourcePlan,
    pub host: AnalysisHostWorkload,
    pub program: PreparedAnalysisProgram,
    pub inputs: Vec<PreparedAnalysisObjects>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ImportedStructuralModalResponseSpectrumDirection {
    pub result: StructuralModalResponseSpectrumDirectionResult,
    pub objects: PreparedAnalysisObjects,
}

pub fn prepare_structural_modal_response_spectrum_fanout(
    preparation: StructuralModalResponseSpectrumFanoutPreparation,
) -> AnalysisBridgeResult<Vec<PreparedStructuralModalResponseSpectrumDirectionTask>> {
    if preparation.chunk_bytes == 0 {
        return Err(AnalysisBridgeError::Invalid(
            "response-spectrum fan-out requires bounded nonzero chunks".into(),
        ));
    }
    let constraints = CanonicalStructuralConstraintPlan::compile_for_model(
        preparation.workload.model(),
        Some(&preparation.mesh),
    )
    .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let prepared = runmat_analysis_fea::prepare_structural_modal_response_spectrum(
        &preparation.workload,
        &preparation.mesh,
        &preparation.modal_state,
        &constraints,
        &[],
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
    let workload_objects = prepare_structural_modal_response_spectrum_workload_objects(
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
    (0..preparation.workload.spectra()?.len())
        .map(|ordinal| {
            let request = StructuralModalResponseSpectrumDirectionRequest::new(
                &preparation.workload,
                prepared.scope().clone(),
                ordinal,
            )?;
            let request_objects =
                prepare_structural_modal_response_spectrum_direction_request_objects(
                    &request,
                    preparation.chunk_bytes,
                    ObjectNamespace::InputValue,
                    preparation.inventory_limits,
                )?;
            let mut numerical_inputs = shared.clone();
            numerical_inputs.push(request_objects);
            let resource_plan = prepared
                .resources()
                .plan_for_direction(
                    &request,
                    prepared_input_resource_facts(&numerical_inputs)?,
                    projected_output_resource_facts(
                        structural_modal_response_spectrum_direction_payload_bytes_upper_bound(
                            &request,
                        )
                        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?,
                    )?,
                    projected_output_set_resource_facts(),
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
            let host = task_host(
                study,
                step,
                execution_plan,
                AnalysisWorkKind::SolveStructuralModalResponseSpectrumDirection,
                &numerical_inputs,
                vec![AnalysisNumericalObjectKind::StructuralModalResponseSpectrumDirectionResult],
                preparation.artifact_access.clone(),
            )?;
            let program = prepare_analysis_program(
                &host,
                preparation.program_revision.clone(),
                &numerical_inputs,
            )?;
            Ok(PreparedStructuralModalResponseSpectrumDirectionTask {
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

pub fn import_structural_modal_response_spectrum_direction(
    source: &impl CacheImport,
    response: ProgramExecutionResponse,
    prepared: &PreparedStructuralModalResponseSpectrumDirectionTask,
    limits: ObjectInventoryLimits,
) -> AnalysisBridgeResult<ImportedStructuralModalResponseSpectrumDirection> {
    response.validate_against(&prepared.program.request)?;
    let output_set = import_analysis_output_set(source, response, &prepared.host)?;
    let [output] = output_set.manifest.outputs.as_slice() else {
        return Err(AnalysisBridgeError::Invalid(
            "response-spectrum direction must return exactly one result".into(),
        ));
    };
    let objects = import_analysis_objects(
        source,
        &output.root,
        AnalysisNumericalObjectKind::StructuralModalResponseSpectrumDirectionResult,
        output.logical_identity,
        output.payload_digest,
        &prepared.host.artifact_access,
        limits,
    )?;
    let result = decode_prepared_structural_modal_response_spectrum_direction_result(&objects)?;
    if result.request() != &prepared.request {
        return Err(AnalysisBridgeError::Identity(
            "response-spectrum direction result differs from its prepared request",
        ));
    }
    Ok(ImportedStructuralModalResponseSpectrumDirection { result, objects })
}

pub struct StructuralModalResponseSpectrumAssemblyPreparation {
    pub authored_study_digest: Digest,
    pub frozen_project_digest: Option<Digest>,
    pub program_revision: ProgramRevision,
    pub artifact_access: ArtifactAccess,
    pub workload: StructuralModalResponseSpectrumWorkload,
    pub mesh: SolverMeshArtifact,
    pub modal_state: CanonicalSemanticState,
    pub entries: Vec<PreparedAnalysisObjects>,
    pub chunk_bytes: usize,
    pub inventory_limits: ObjectInventoryLimits,
}

pub struct PreparedStructuralModalResponseSpectrumAssemblyTask {
    pub workload: StructuralModalResponseSpectrumWorkload,
    pub request: StructuralModalResponseSpectrumAssemblyRequest,
    pub resource_plan: FeaResourcePlan,
    pub host: AnalysisHostWorkload,
    pub program: PreparedAnalysisProgram,
    pub inputs: Vec<PreparedAnalysisObjects>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ImportedStructuralModalResponseSpectrumResult {
    pub identity: SolverResultIdentity,
    pub result: StructuralModalResponseSpectrumSolverResult,
    pub result_objects: PreparedAnalysisObjects,
    pub numerical_receipt: StructuralModalResponseSpectrumNumericalReceipt,
    pub receipt_objects: PreparedAnalysisObjects,
    pub field_recovery: runmat_analysis_fea::solve::dynamics::ResponseSpectrumFieldRecovery,
    pub field_recovery_objects: PreparedAnalysisObjects,
    pub response: ProgramExecutionResponse,
}

pub fn prepare_structural_modal_response_spectrum_assembly(
    preparation: StructuralModalResponseSpectrumAssemblyPreparation,
) -> AnalysisBridgeResult<PreparedStructuralModalResponseSpectrumAssemblyTask> {
    if preparation.chunk_bytes == 0 {
        return Err(AnalysisBridgeError::Invalid(
            "response-spectrum fan-in requires bounded nonzero chunks".into(),
        ));
    }
    let entries = preparation
        .entries
        .iter()
        .map(decode_prepared_structural_modal_response_spectrum_direction_result)
        .collect::<AnalysisBridgeResult<Vec<_>>>()?;
    let constraints = CanonicalStructuralConstraintPlan::compile_for_model(
        preparation.workload.model(),
        Some(&preparation.mesh),
    )
    .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let prepared = runmat_analysis_fea::prepare_structural_modal_response_spectrum(
        &preparation.workload,
        &preparation.mesh,
        &preparation.modal_state,
        &constraints,
        &[],
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
    let request =
        StructuralModalResponseSpectrumAssemblyRequest::new(prepared.scope().clone(), entries)?;
    let workload_objects = prepare_structural_modal_response_spectrum_workload_objects(
        &preparation.workload,
        preparation.chunk_bytes,
        ObjectNamespace::InputValue,
        preparation.inventory_limits,
    )?;
    let request_objects = prepare_structural_modal_response_spectrum_assembly_request_objects(
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
    numerical_inputs.extend(preparation.entries);
    let resource_plan = prepared
        .resources()
        .plan_for_assembly(
            &request,
            prepared_input_resource_facts(&numerical_inputs)?,
            projected_output_resource_facts(
                structural_modal_response_spectrum_assembly_payload_bytes_upper_bound(&request)
                    .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?,
            )?,
            projected_output_resource_facts(
                structural_modal_response_spectrum_receipt_payload_bytes_upper_bound(),
            )?,
            projected_output_resource_facts(
                prepared
                    .resources()
                    .field_recovery_payload_bytes_upper_bound()
                    .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?,
            )?,
            projected_output_set_resource_facts(),
        )
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    numerical_inputs.push(resource_plan_objects(
        &resource_plan,
        preparation.chunk_bytes,
        preparation.inventory_limits,
    )?);
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
    let host = task_host(
        study,
        step,
        execution_plan,
        AnalysisWorkKind::AssembleStructuralModalResponseSpectrumResult,
        &numerical_inputs,
        vec![
            AnalysisNumericalObjectKind::SolverResult,
            AnalysisNumericalObjectKind::NumericalReceipt,
            AnalysisNumericalObjectKind::StructuralModalResponseSpectrumFieldRecovery,
        ],
        preparation.artifact_access,
    )?;
    let program = prepare_analysis_program(&host, preparation.program_revision, &numerical_inputs)?;
    Ok(PreparedStructuralModalResponseSpectrumAssemblyTask {
        workload: preparation.workload,
        request,
        resource_plan,
        host,
        program,
        inputs: numerical_inputs,
    })
}

pub fn import_structural_modal_response_spectrum_result(
    source: &impl CacheImport,
    response: ProgramExecutionResponse,
    prepared: &PreparedStructuralModalResponseSpectrumAssemblyTask,
    limits: ObjectInventoryLimits,
) -> AnalysisBridgeResult<ImportedStructuralModalResponseSpectrumResult> {
    response.validate_against(&prepared.program.request)?;
    let retained_response = response.clone();
    let output_set = import_analysis_output_set(source, response, &prepared.host)?;
    let [result_output, receipt_output, field_recovery_output] =
        output_set.manifest.outputs.as_slice()
    else {
        return Err(AnalysisBridgeError::Invalid(
            "response-spectrum fan-in must return one result, one receipt, and one field-recovery result"
                .into(),
        ));
    };
    let identity = SolverResultIdentity::new(
        prepared.host.workload.identity,
        result_output.payload_digest,
    )?;
    if result_output.logical_identity != identity.digest() {
        return Err(AnalysisBridgeError::Identity(
            "response-spectrum result root differs from its workload-bound identity",
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
    let field_recovery_objects = import_analysis_objects(
        source,
        &field_recovery_output.root,
        AnalysisNumericalObjectKind::StructuralModalResponseSpectrumFieldRecovery,
        field_recovery_output.logical_identity,
        field_recovery_output.payload_digest,
        &prepared.host.artifact_access,
        limits,
    )?;
    let result = decode_prepared_structural_modal_response_spectrum_solver_result(&result_objects)?;
    result.validate_for(&prepared.workload)?;
    if result.assembly() != &prepared.request {
        return Err(AnalysisBridgeError::Identity(
            "response-spectrum result differs from its prepared fan-in",
        ));
    }
    let numerical_receipt =
        decode_prepared_structural_modal_response_spectrum_numerical_receipt(&receipt_objects)?;
    numerical_receipt.validate_against(identity, &result)?;
    if receipt_output.logical_identity != numerical_receipt.identity().digest() {
        return Err(AnalysisBridgeError::Identity(
            "response-spectrum receipt root differs from its canonical identity",
        ));
    }
    let field_recovery =
        decode_prepared_structural_modal_response_spectrum_field_recovery(&field_recovery_objects)?;
    field_recovery
        .validate_for(result.numerical())
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    if field_recovery_output.logical_identity
        != field_recovery
            .identity()
            .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?
    {
        return Err(AnalysisBridgeError::Identity(
            "response-spectrum field-recovery root differs from its canonical identity",
        ));
    }
    let mut expected = result_objects
        .unique_references(&prepared.host.artifact_access, ValueRefKind::ResultObject)?;
    expected.extend(
        receipt_objects
            .unique_references(&prepared.host.artifact_access, ValueRefKind::ResultObject)?,
    );
    expected.extend(
        field_recovery_objects
            .unique_references(&prepared.host.artifact_access, ValueRefKind::ResultObject)?,
    );
    crate::object::retain_unique_references(&mut expected);
    expected.push(output_set.root);
    if output_set.result_objects != expected {
        return Err(AnalysisBridgeError::Identity(
            "response-spectrum result inventory differs from its canonical closure",
        ));
    }
    Ok(ImportedStructuralModalResponseSpectrumResult {
        identity,
        result,
        result_objects,
        numerical_receipt,
        receipt_objects,
        field_recovery,
        field_recovery_objects,
        response: retained_response,
    })
}

fn execution_plan(
    workload: &StructuralModalResponseSpectrumWorkload,
    scope: &runmat_analysis_fea::execution::StructuralModalResponseSpectrumScope,
    constraint_identity: ConstraintPlanIdentity,
    step: StepIdentity,
    topology_digest: Digest,
    resources: &FeaResourcePlan,
) -> AnalysisBridgeResult<ExecutionPlanIdentity> {
    let problem = ProblemIdentity::new(
        step,
        contract_digest(
            "response-spectrum-model",
            &workload.model().solver_identity_contract(),
        )?,
        Some(workload.modal_basis_transition().identity),
        constraint_identity,
        Vec::new(),
    )?;
    let discretization = DiscretizationIdentity::new(
        problem,
        topology_digest,
        contract_digest(
            "response-spectrum-formulation",
            &workload.formulation_requirement(),
        )?,
    )?;
    let operator = OperatorIdentity::new(discretization, scope.operator_digest())?;
    let numerical = NumericalPlanIdentity::new(
        operator,
        contract_digest(
            "response-spectrum-solver",
            &(
                scope.basis(),
                scope.response(),
                scope.directional_combination(),
                workload.backend(),
            ),
        )?,
    )?;
    ExecutionPlanIdentity::new(
        numerical,
        contract_digest("response-spectrum-execution", &resources.identity)?,
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
