use std::collections::BTreeSet;

use runmat_analysis_fea::{
    constraints::CanonicalStructuralConstraintPlan,
    execution::{
        AnalysisNumericalObjectKind, AnalysisWorkKind, CanonicalSemanticState,
        ConstraintPlanIdentity, DiscretizationIdentity, ExecutionPlanIdentity,
        NumericalPlanIdentity, OperatorIdentity, ProblemIdentity, SequenceIdentity,
        SolverResultIdentity, StepIdentity, StructuralBucklingNumericalReceipt,
        StructuralBucklingSolverResult, StructuralBucklingWorkload, StudyIdentity,
    },
    planning::resource::{
        build_structural_buckling_resource_plan,
        structural_buckling_receipt_payload_bytes_upper_bound,
        structural_buckling_result_payload_bytes_upper_bound, FeaResourcePlan,
        PreparedStructuralBucklingResourceFacts,
    },
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
    decode_prepared_structural_buckling_receipt, decode_prepared_structural_buckling_result,
    decode_prepared_structural_buckling_workload, import_analysis_objects,
    import_analysis_output_set, prepare_analysis_objects, prepare_analysis_program,
    prepare_semantic_state_objects, prepare_structural_buckling_workload_objects,
    prepared_input_resource_facts, projected_output_resource_facts,
    projected_output_set_resource_facts, AnalysisBridgeError, AnalysisBridgeResult,
    AnalysisCapabilityRequirement, AnalysisHostWorkload, AnalysisWorkloadInput,
    AnalysisWorkloadRequest, PreparedAnalysisObjects, PreparedAnalysisProgram, ANALYSIS_HOST_ABI,
};

const BUCKLING_TASK_PREFIX: &[u8] = b"runmat-analysis-structural-buckling-task/v1\0";
const BUCKLING_TASK_LIMITS: CanonicalLimits =
    CanonicalLimits::new(64 * 1024 * 1024, 1_000_000, 65_536, 64);

pub struct StructuralBucklingTaskPreparation {
    pub local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan,
    pub authored_study_digest: Digest,
    pub frozen_project_digest: Option<Digest>,
    pub program_revision: ProgramRevision,
    pub artifact_access: ArtifactAccess,
    pub workload: StructuralBucklingWorkload,
    pub mesh: SolverMeshArtifact,
    pub prestress_state: CanonicalSemanticState,
    pub chunk_bytes: usize,
    pub inventory_limits: ObjectInventoryLimits,
    pub required_capabilities: BTreeSet<AnalysisCapabilityRequirement>,
    pub accelerator_requests: Vec<AcceleratorRequest>,
}

pub struct PreparedStructuralBucklingTask {
    pub host: AnalysisHostWorkload,
    pub program: PreparedAnalysisProgram,
    pub inputs: Vec<PreparedAnalysisObjects>,
    pub resource_plan: FeaResourcePlan,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ImportedStructuralBucklingResult {
    pub host: AnalysisHostWorkload,
    pub identity: SolverResultIdentity,
    pub result: StructuralBucklingSolverResult,
    pub objects: PreparedAnalysisObjects,
    pub numerical_receipt: StructuralBucklingNumericalReceipt,
    pub numerical_receipt_objects: PreparedAnalysisObjects,
    pub response: ProgramExecutionResponse,
}

pub fn prepare_structural_buckling_task(
    preparation: StructuralBucklingTaskPreparation,
) -> AnalysisBridgeResult<PreparedStructuralBucklingTask> {
    preparation.workload.validate()?;
    preparation.prestress_state.validate()?;
    preparation
        .mesh
        .validate_canonical()
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let topology_digest = Digest::from_bytes(*preparation.mesh.canonical_digest.bytes());
    let linearization = preparation
        .prestress_state
        .payload
        .structural_linearization()
        .ok_or_else(|| {
            AnalysisBridgeError::Invalid("buckling prestress payload is absent".into())
        })?;
    if preparation.chunk_bytes == 0
        || preparation.mesh.geometry.geometry_revision
            != u64::from(preparation.workload.model().geometry_revision)
        || preparation.prestress_state.identity
            != preparation.workload.prestress_transition().source_state
        || preparation.prestress_state.topology_digest != topology_digest
        || linearization.topology_digest() != topology_digest
    {
        return Err(AnalysisBridgeError::Invalid(
            "buckling task has mismatched mesh, workload, or prestress state".into(),
        ));
    }
    let constraints = CanonicalStructuralConstraintPlan::compile_for_model(
        preparation.workload.model(),
        Some(&preparation.mesh),
    )
    .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let workload_objects = prepare_structural_buckling_workload_objects(
        &preparation.workload,
        preparation.chunk_bytes,
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
        &preparation.prestress_state,
        preparation.chunk_bytes,
        ObjectNamespace::InputValue,
        preparation.inventory_limits,
    )?;
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
    let numerical_inputs = vec![
        workload_objects,
        mesh_objects,
        state_objects,
        constraint_objects,
    ];
    let result_bound = structural_buckling_result_payload_bytes_upper_bound(
        linearization.base_displacement().len(),
        preparation.workload.solver().requested_mode_count(),
    )
    .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let resource_plan = build_structural_buckling_resource_plan(
        &preparation.mesh,
        &constraints,
        &preparation.workload,
        &preparation.prestress_state,
        &preparation.accelerator_requests,
        PreparedStructuralBucklingResourceFacts {
            local_execution: preparation.local_execution,
            inputs: prepared_input_resource_facts(&numerical_inputs)?,
            result_output: projected_output_resource_facts(result_bound)?,
            receipt_output: projected_output_resource_facts(
                structural_buckling_receipt_payload_bytes_upper_bound(),
            )?,
            output_set: projected_output_set_resource_facts(),
        },
    )
    .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let resource_objects = prepare_analysis_objects(
        AnalysisNumericalObjectKind::ResourcePlan,
        resource_plan.identity.digest(),
        &resource_plan
            .canonical_bytes()
            .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?,
        preparation.chunk_bytes,
        ObjectNamespace::InputValue,
        preparation.inventory_limits,
    )?;
    let problem = ProblemIdentity::new(
        step,
        contract_digest(
            "buckling-model",
            &preparation.workload.model().solver_identity_contract(),
        )?,
        Some(preparation.workload.prestress_transition().identity),
        constraint_identity,
        Vec::new(),
    )?;
    let discretization = DiscretizationIdentity::new(
        problem,
        topology_digest,
        contract_digest("buckling-formulation", &topology_digest)?,
    )?;
    let operator = OperatorIdentity::new(
        discretization,
        resource_plan
            .input
            .structural_operator()
            .ok_or_else(|| AnalysisBridgeError::Invalid("buckling operator plan is absent".into()))?
            .numerical_scope_digest,
    )?;
    let numerical_plan = NumericalPlanIdentity::new(
        operator,
        contract_digest(
            "buckling-solver",
            &(
                preparation.workload.solver(),
                preparation.workload.backend(),
            ),
        )?,
    )?;
    let execution_plan = ExecutionPlanIdentity::new(
        numerical_plan,
        contract_digest("buckling-execution", &resource_plan.identity)?,
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
    let request = AnalysisWorkloadRequest::new(
        StudyIdentity::from_authored_digest(
            preparation.authored_study_digest,
            preparation.frozen_project_digest,
        )?,
        step,
        execution_plan,
        AnalysisWorkKind::SolveStructuralBucklingModes,
        workload_inputs,
        vec![
            AnalysisNumericalObjectKind::SolverResult,
            AnalysisNumericalObjectKind::NumericalReceipt,
        ],
        required_capabilities,
    )?;
    let host = AnalysisHostWorkload::new(
        request,
        preparation.artifact_access,
        crate::AnalysisOutputPolicy::new(preparation.chunk_bytes, preparation.inventory_limits)?,
    )?;
    let program = prepare_analysis_program(&host, preparation.program_revision, &inputs)?;
    Ok(PreparedStructuralBucklingTask {
        host,
        program,
        inputs,
        resource_plan,
    })
}

pub fn import_structural_buckling_result(
    source: &impl CacheImport,
    response: ProgramExecutionResponse,
    prepared: &PreparedStructuralBucklingTask,
    limits: ObjectInventoryLimits,
) -> AnalysisBridgeResult<ImportedStructuralBucklingResult> {
    response.validate_against(&prepared.program.request)?;
    let retained_response = response.clone();
    let output_set = import_analysis_output_set(source, response, &prepared.host)?;
    let [result_output, receipt_output] = output_set.manifest.outputs.as_slice() else {
        return Err(AnalysisBridgeError::Invalid(
            "buckling task must return one result and one numerical receipt".into(),
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
            "buckling result inventory differs from its canonical closure",
        ));
    }
    let workload = prepared
        .inputs
        .iter()
        .find(|input| {
            input.manifest.kind == AnalysisNumericalObjectKind::StructuralBucklingWorkload
        })
        .ok_or_else(|| {
            AnalysisBridgeError::Invalid("prepared buckling workload is absent".into())
        })?;
    let workload = decode_prepared_structural_buckling_workload(workload)?;
    let result = decode_prepared_structural_buckling_result(&objects)?;
    result.validate_for(&workload)?;
    let numerical_receipt = decode_prepared_structural_buckling_receipt(&receipt_objects)?;
    numerical_receipt.validate_against(identity, &result)?;
    Ok(ImportedStructuralBucklingResult {
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
        BUCKLING_TASK_PREFIX,
        domain,
        value,
        BUCKLING_TASK_LIMITS,
    )?))
}
