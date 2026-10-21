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
    pub local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan,
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
                work_kind: AnalysisWorkKind::SolveStructuralModalResponseSpectrumDirection,
                inputs: &numerical_inputs,
                outputs: vec![
                    AnalysisNumericalObjectKind::StructuralModalResponseSpectrumDirectionResult,
                ],
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
    pub local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan,
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
    pub resource_plan_objects: PreparedAnalysisObjects,
    pub host: AnalysisHostWorkload,
    pub program: PreparedAnalysisProgram,
    pub inputs: Vec<PreparedAnalysisObjects>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ImportedStructuralModalResponseSpectrumResult {
    pub state_publication_profiles: runmat_analysis_fea::execution::StatePublicationProfiles,
    pub state_publication_profile_objects: PreparedAnalysisObjects,
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
            runmat_analysis_fea::planning::resource::PreparedStructuralResponseSpectrumAssemblyOutputs {
            result: projected_output_resource_facts(
                structural_modal_response_spectrum_assembly_payload_bytes_upper_bound(&request)
                    .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?,
            )?,
            receipt: projected_output_resource_facts(
                structural_modal_response_spectrum_receipt_payload_bytes_upper_bound(),
            )?,
            field_recovery: projected_output_resource_facts(
                prepared
                    .resources()
                    .field_recovery_payload_bytes_upper_bound()
                    .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?,
            )?,
            state_profiles: crate::structural_state_profiles_output_facts(
                preparation.workload.model(), preparation.workload.procedure_step()?,
                &[runmat_analysis_fea::execution::SemanticStatePublicationSource::StructuralResponseEnvelope], request.scope().topology_digest(),
            )?,
            },
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
        work_kind: AnalysisWorkKind::AssembleStructuralModalResponseSpectrumResult,
        inputs: &numerical_inputs,
        outputs: vec![
            AnalysisNumericalObjectKind::SolverResult,
            AnalysisNumericalObjectKind::NumericalReceipt,
            AnalysisNumericalObjectKind::StructuralModalResponseSpectrumFieldRecovery,
            AnalysisNumericalObjectKind::StatePublicationProfiles,
        ],
        artifact_access: preparation.artifact_access,
        output_policy: crate::AnalysisOutputPolicy::new(
            preparation.chunk_bytes,
            preparation.inventory_limits,
        )?,
    })?;
    let program = prepare_analysis_program(&host, preparation.program_revision, &numerical_inputs)?;
    Ok(PreparedStructuralModalResponseSpectrumAssemblyTask {
        workload: preparation.workload,
        request,
        resource_plan,
        resource_plan_objects,
        host,
        program,
        inputs: numerical_inputs,
    })
}

mod import;
pub use import::import_structural_modal_response_spectrum_result;

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
        contract_digest("response-spectrum-formulation", &topology_digest)?,
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
