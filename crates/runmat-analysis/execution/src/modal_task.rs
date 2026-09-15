use std::collections::BTreeSet;

use runmat_analysis_fea::{
    constraints::CanonicalStructuralConstraintPlan,
    execution::{
        AnalysisNumericalObjectKind, AnalysisWorkKind, ConstraintPlanIdentity,
        DiscretizationIdentity, ExecutionPlanIdentity, NumericalPlanIdentity, OperatorIdentity,
        ProblemIdentity, SequenceIdentity, SolverResultIdentity, StepIdentity,
        StructuralModalNumericalReceipt, StructuralModalSolverResult, StructuralModalWorkload,
        StudyIdentity,
    },
    planning::resource::{
        build_structural_modal_resource_plan, structural_modal_receipt_payload_bytes_upper_bound,
        structural_modal_result_payload_bytes_upper_bound, FeaResourcePlan,
        PreparedStructuralModalResourceFacts,
    },
    post::fields::structural_observable_field_definitions,
    results::observation::{resolve_structural_observation_plan, ObservationLimits},
};
use runmat_canonical_codec::{encode_contract, CanonicalLimits};
use runmat_execution::{
    resource::AcceleratorRequest, value::ValueRefKind, Digest, ProgramRevision,
};
use runmat_execution_artifact::{
    cache::CacheImport, object::ObjectInventoryLimits, ArtifactAccess, ObjectNamespace,
    ProgramExecutionResponse,
};
use runmat_meshing_core::{CanonicalMeshingContract, SolverMeshArtifact};
use serde::Serialize;

use crate::{
    decode_prepared_structural_modal_receipt, decode_prepared_structural_modal_result,
    decode_prepared_structural_modal_workload, import_analysis_objects, import_analysis_output_set,
    prepare_analysis_objects, prepare_analysis_program, prepare_structural_modal_workload_objects,
    prepared_input_resource_facts, projected_output_resource_facts,
    projected_output_set_resource_facts, AnalysisBridgeError, AnalysisBridgeResult,
    AnalysisCapabilityRequirement, AnalysisHostWorkload, AnalysisWorkloadInput,
    AnalysisWorkloadRequest, PreparedAnalysisObjects, PreparedAnalysisProgram, ANALYSIS_HOST_ABI,
};

const MODAL_TASK_PREFIX: &[u8] = b"runmat-analysis-structural-modal-task/v1\0";
const MODAL_TASK_LIMITS: CanonicalLimits =
    CanonicalLimits::new(64 * 1024 * 1024, 1_000_000, 65_536, 64);

pub struct StructuralModalTaskPreparation {
    pub authored_study_digest: Digest,
    pub frozen_project_digest: Option<Digest>,
    pub program_revision: ProgramRevision,
    pub artifact_access: ArtifactAccess,
    pub workload: StructuralModalWorkload,
    pub mesh: SolverMeshArtifact,
    pub chunk_bytes: usize,
    pub inventory_limits: ObjectInventoryLimits,
    pub required_capabilities: BTreeSet<AnalysisCapabilityRequirement>,
    pub accelerator_requests: Vec<AcceleratorRequest>,
}

pub struct PreparedStructuralModalTask {
    pub host: AnalysisHostWorkload,
    pub program: PreparedAnalysisProgram,
    pub inputs: Vec<PreparedAnalysisObjects>,
    pub resource_plan: FeaResourcePlan,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ImportedStructuralModalResult {
    pub host: AnalysisHostWorkload,
    pub identity: SolverResultIdentity,
    pub result: StructuralModalSolverResult,
    pub objects: PreparedAnalysisObjects,
    pub numerical_receipt: StructuralModalNumericalReceipt,
    pub numerical_receipt_objects: PreparedAnalysisObjects,
    pub response: ProgramExecutionResponse,
}

pub fn prepare_structural_modal_task(
    preparation: StructuralModalTaskPreparation,
) -> AnalysisBridgeResult<PreparedStructuralModalTask> {
    preparation.workload.validate()?;
    preparation
        .mesh
        .validate_canonical()
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    if preparation.chunk_bytes == 0
        || preparation.mesh.geometry.geometry_revision
            != u64::from(preparation.workload.model().geometry_revision)
    {
        return Err(AnalysisBridgeError::Invalid(
            "modal task has an invalid chunk size or mismatched mesh revision".into(),
        ));
    }
    let constraints = CanonicalStructuralConstraintPlan::compile_for_model(
        preparation.workload.model(),
        Some(&preparation.mesh),
    )
    .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let field_definitions = structural_observable_field_definitions()
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let observations = resolve_structural_observation_plan(
        preparation.workload.procedure_step()?,
        preparation.workload.model(),
        Some(&preparation.mesh),
        &constraints,
        &field_definitions,
        ObservationLimits::default(),
    )
    .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let workload_objects = prepare_structural_modal_workload_objects(
        &preparation.workload,
        preparation.chunk_bytes,
        preparation.inventory_limits,
    )?;
    let mesh_bytes = preparation
        .mesh
        .canonical_encode()
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let mesh_digest = Digest::from_bytes(*preparation.mesh.canonical_digest.bytes());
    let mesh_objects = prepare_analysis_objects(
        AnalysisNumericalObjectKind::Mesh,
        mesh_digest,
        &mesh_bytes,
        preparation.chunk_bytes,
        ObjectNamespace::InputValue,
        preparation.inventory_limits,
    )?;
    let constraint_bytes = constraints
        .canonical_bytes()
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let sequence = SequenceIdentity::from_solver_contract(&preparation.workload.model().sequence)?;
    let step =
        StepIdentity::from_solver_contract(sequence, preparation.workload.procedure_step()?)?;
    let constraint_identity = ConstraintPlanIdentity::new(
        step,
        mesh_digest,
        constraints
            .content_digest()
            .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?,
    )?;
    let constraint_objects = prepare_analysis_objects(
        AnalysisNumericalObjectKind::ConstraintPlan,
        constraint_identity.digest(),
        &constraint_bytes,
        preparation.chunk_bytes,
        ObjectNamespace::InputValue,
        preparation.inventory_limits,
    )?;
    let numerical_inputs = vec![workload_objects, mesh_objects, constraint_objects];
    let result_payload_upper_bound = structural_modal_result_payload_bytes_upper_bound(
        Some(&preparation.mesh),
        &constraints,
        &preparation.workload,
    )
    .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let resource_plan = build_structural_modal_resource_plan(
        Some(&preparation.mesh),
        &constraints,
        &observations,
        &preparation.workload,
        &preparation.accelerator_requests,
        PreparedStructuralModalResourceFacts {
            inputs: prepared_input_resource_facts(&numerical_inputs)?,
            result_output: projected_output_resource_facts(result_payload_upper_bound)?,
            receipt_output: projected_output_resource_facts(
                structural_modal_receipt_payload_bytes_upper_bound(),
            )?,
            output_set: projected_output_set_resource_facts(),
        },
    )
    .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let resource_bytes = resource_plan
        .canonical_bytes()
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let resource_objects = prepare_analysis_objects(
        AnalysisNumericalObjectKind::ResourcePlan,
        resource_plan.identity.digest(),
        &resource_bytes,
        preparation.chunk_bytes,
        ObjectNamespace::InputValue,
        preparation.inventory_limits,
    )?;
    let study = StudyIdentity::from_authored_digest(
        preparation.authored_study_digest,
        preparation.frozen_project_digest,
    )?;
    let problem = ProblemIdentity::new(
        step,
        contract_digest(
            "modal-model",
            &preparation.workload.model().solver_identity_contract(),
        )?,
        None,
        constraint_identity,
        Vec::new(),
    )?;
    let discretization = DiscretizationIdentity::new(
        problem,
        mesh_digest,
        contract_digest(
            "modal-formulation",
            &preparation.workload.formulation_requirement(),
        )?,
    )?;
    let operator_scope = resource_plan
        .input
        .structural_operator()
        .ok_or_else(|| AnalysisBridgeError::Invalid("modal resource plan has no operator".into()))?
        .numerical_scope_digest;
    let operator = OperatorIdentity::new(discretization, operator_scope)?;
    let numerical_plan = NumericalPlanIdentity::new(
        operator,
        contract_digest(
            "modal-solver",
            &(
                preparation.workload.truncation(),
                preparation.workload.convergence(),
                preparation.workload.backend(),
            ),
        )?,
    )?;
    let execution_plan = ExecutionPlanIdentity::new(
        numerical_plan,
        contract_digest("modal-execution", &resource_plan.identity)?,
    )?;
    let mut inputs = numerical_inputs;
    inputs.push(resource_objects);
    let workload_inputs = inputs
        .iter()
        .map(|input| AnalysisWorkloadInput {
            kind: input.manifest.kind,
            logical_identity: input.manifest.logical_identity,
            payload_digest: input.manifest.payload_digest,
            manifest_digest: input.root.digest,
            manifest_encoded_length: input.root.encoded_length,
            reference_kind: ValueRefKind::DriverObject,
        })
        .collect();
    let mut required_capabilities = preparation.required_capabilities;
    required_capabilities.insert(AnalysisCapabilityRequirement::Host {
        abi: ANALYSIS_HOST_ABI.into(),
    });
    for accelerator in &resource_plan.request.accelerators {
        required_capabilities.insert(AnalysisCapabilityRequirement::Accelerator {
            class: accelerator.class.clone(),
        });
    }
    let request = AnalysisWorkloadRequest::new(
        study,
        step,
        execution_plan,
        AnalysisWorkKind::ExtractStructuralModes,
        workload_inputs,
        vec![
            AnalysisNumericalObjectKind::SolverResult,
            AnalysisNumericalObjectKind::NumericalReceipt,
        ],
        required_capabilities,
    )?;
    let host = AnalysisHostWorkload::new(request, preparation.artifact_access)?;
    let program = prepare_analysis_program(&host, preparation.program_revision, &inputs)?;
    Ok(PreparedStructuralModalTask {
        host,
        program,
        inputs,
        resource_plan,
    })
}

pub fn import_structural_modal_result(
    source: &impl CacheImport,
    response: ProgramExecutionResponse,
    prepared: &PreparedStructuralModalTask,
    limits: ObjectInventoryLimits,
) -> AnalysisBridgeResult<ImportedStructuralModalResult> {
    response.validate_against(&prepared.program.request)?;
    let retained_response = response.clone();
    let output_set = import_analysis_output_set(source, response, &prepared.host)?;
    let [result_output, receipt_output] = output_set.manifest.outputs.as_slice() else {
        return Err(AnalysisBridgeError::Invalid(
            "modal extraction must return one result and one numerical receipt".into(),
        ));
    };
    let identity = SolverResultIdentity::new(
        prepared.host.workload.identity,
        result_output.payload_digest,
    )?;
    let objects = import_analysis_objects(
        source,
        &result_output.root,
        AnalysisNumericalObjectKind::SolverResult,
        identity.digest(),
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
        objects.unique_references(&prepared.host.artifact_access, ValueRefKind::ResultObject)?;
    expected.extend(
        receipt_objects
            .unique_references(&prepared.host.artifact_access, ValueRefKind::ResultObject)?,
    );
    crate::object::retain_unique_references(&mut expected);
    expected.push(output_set.root);
    if output_set.result_objects != expected {
        return Err(AnalysisBridgeError::Identity(
            "modal result inventory differs from its canonical closure",
        ));
    }
    let workload = prepared
        .inputs
        .iter()
        .find(|input| input.manifest.kind == AnalysisNumericalObjectKind::StructuralModalWorkload)
        .ok_or_else(|| AnalysisBridgeError::Invalid("prepared modal workload is absent".into()))?;
    let workload = decode_prepared_structural_modal_workload(workload)?;
    let result = decode_prepared_structural_modal_result(&objects)?;
    result.validate_for(&workload)?;
    let mesh = prepared
        .inputs
        .iter()
        .find(|input| input.manifest.kind == AnalysisNumericalObjectKind::Mesh)
        .ok_or_else(|| AnalysisBridgeError::Invalid("prepared modal mesh is absent".into()))?;
    if result.topology_digest() != mesh.manifest.logical_identity {
        return Err(AnalysisBridgeError::Identity(
            "modal result topology differs from its prepared mesh",
        ));
    }
    let numerical_receipt = decode_prepared_structural_modal_receipt(&receipt_objects)?;
    if numerical_receipt.identity().digest() != receipt_output.logical_identity {
        return Err(AnalysisBridgeError::Identity(
            "modal receipt manifest differs from its canonical identity",
        ));
    }
    numerical_receipt.validate_against(identity, &result)?;
    Ok(ImportedStructuralModalResult {
        host: prepared.host.clone(),
        identity,
        result,
        objects,
        numerical_receipt,
        numerical_receipt_objects: receipt_objects,
        response: retained_response,
    })
}

fn contract_digest<T: Serialize>(domain: &'static str, value: &T) -> AnalysisBridgeResult<Digest> {
    Ok(Digest::sha256(encode_contract(
        MODAL_TASK_PREFIX,
        domain,
        value,
        MODAL_TASK_LIMITS,
    )?))
}
