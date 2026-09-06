use std::collections::BTreeSet;

use runmat_analysis_fea::execution::{
    AccuracyIdentity, AnalysisNumericalObjectKind, AnalysisObjectManifest, AnalysisWorkKind,
    ConstraintPlanIdentity, DiscretizationIdentity, ExecutionPlanIdentity, NumericalPlanIdentity,
    OperatorIdentity, ProblemIdentity, SequenceIdentity, SignalMaterializationIdentity,
    SignalRealizationIdentity, SolverExcitationIdentity, StepIdentity, StructuralLinearWorkload,
    StructuralLinearWorkloadResult, StudyIdentity,
};
use runmat_canonical_codec::{encode_contract, CanonicalLimits};
use runmat_execution::value::{ValuePayload, ValueRefKind};
use runmat_execution::{Digest, ProgramRevision};
use runmat_execution_artifact::cache::CacheImport;
use runmat_execution_artifact::object::ObjectInventoryLimits;
use runmat_execution_artifact::{ArtifactAccess, ObjectNamespace, ProgramExecutionResponse};
use runmat_meshing_core::{CanonicalMeshingContract, SolverMeshArtifact};
use serde::Serialize;

use crate::{
    import_analysis_objects, prepare_analysis_objects, prepare_analysis_program,
    AnalysisBridgeError, AnalysisBridgeResult, AnalysisCapabilityRequirement, AnalysisHostWorkload,
    AnalysisWorkloadInput, AnalysisWorkloadRequest, PreparedAnalysisObjects,
    PreparedAnalysisProgram, ANALYSIS_HOST_ABI,
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
    pub chunk_bytes: usize,
    pub inventory_limits: ObjectInventoryLimits,
    pub required_capabilities: BTreeSet<AnalysisCapabilityRequirement>,
}

pub struct PreparedStructuralLinearProgram {
    pub host: AnalysisHostWorkload,
    pub program: PreparedAnalysisProgram,
    pub inputs: Vec<PreparedAnalysisObjects>,
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

    let solve_bytes = preparation.workload.canonical_bytes()?;
    let mesh_bytes = preparation
        .mesh
        .canonical_encode()
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let mesh_digest = Digest::from_bytes(*preparation.mesh.canonical_digest.bytes());
    let identities = derive_identities(
        &preparation.workload,
        preparation.authored_study_digest,
        preparation.frozen_project_digest,
        mesh_digest,
    )?;
    let state = prepare_analysis_objects(
        AnalysisNumericalObjectKind::SemanticState,
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
    let inputs = vec![state, mesh];
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
    let workload = AnalysisWorkloadRequest::new(
        identities.study,
        identities.step,
        identities.execution_plan,
        AnalysisWorkKind::Solve,
        workload_inputs,
        vec![AnalysisNumericalObjectKind::SemanticState],
        required_capabilities,
    )?;
    let host = AnalysisHostWorkload::new(workload, preparation.artifact_access)?;
    let program = prepare_analysis_program(&host, preparation.program_revision, &inputs)?;
    Ok(PreparedStructuralLinearProgram {
        host,
        program,
        inputs,
    })
}

pub fn import_structural_linear_result(
    source: &impl CacheImport,
    response: ProgramExecutionResponse,
    prepared: &PreparedStructuralLinearProgram,
    limits: ObjectInventoryLimits,
) -> AnalysisBridgeResult<StructuralLinearWorkloadResult> {
    response.validate_against(&prepared.program.request)?;
    let ProgramExecutionResponse::ExternalizedSuccess {
        outputs,
        result_objects,
    } = response
    else {
        return Err(AnalysisBridgeError::Invalid(
            "structural analysis did not return an externalized success".into(),
        ));
    };
    let [ValuePayload::Object(root)] = outputs.as_slice() else {
        return Err(AnalysisBridgeError::Invalid(
            "structural analysis must return one externalized result root".into(),
        ));
    };
    let manifest_bytes = source
        .read_verified(root.logical_digest)?
        .ok_or_else(|| AnalysisBridgeError::Invalid("analysis result manifest is absent".into()))?;
    let manifest = AnalysisObjectManifest::from_canonical_bytes(&manifest_bytes)?;
    let imported = import_analysis_objects(
        source,
        root,
        AnalysisNumericalObjectKind::SemanticState,
        prepared.host.workload.execution_plan.digest(),
        manifest.payload_digest,
        &prepared.host.artifact_access,
        limits,
    )?;
    let expected =
        imported.references(&prepared.host.artifact_access, ValueRefKind::ResultObject)?;
    if result_objects != expected {
        return Err(AnalysisBridgeError::Identity(
            "analysis result inventory differs from its canonical closure",
        ));
    }
    let result = StructuralLinearWorkloadResult::from_canonical_bytes(&imported.payload())?;
    validate_result_semantics(&prepared.inputs[0], &result)?;
    Ok(result)
}

struct StructuralIdentities {
    study: StudyIdentity,
    step: StepIdentity,
    problem: ProblemIdentity,
    execution_plan: ExecutionPlanIdentity,
}

fn derive_identities(
    workload: &StructuralLinearWorkload,
    authored_study_digest: Digest,
    frozen_project_digest: Option<Digest>,
    mesh_digest: Digest,
) -> AnalysisBridgeResult<StructuralIdentities> {
    let step_contract = workload.model.sequence.steps.first().ok_or_else(|| {
        AnalysisBridgeError::Invalid("structural workload has no procedure step".into())
    })?;
    if workload.model.sequence.steps.len() != 1 {
        return Err(AnalysisBridgeError::Invalid(
            "structural portable execution currently requires exactly one procedure step".into(),
        ));
    }
    let study = StudyIdentity::new(authored_study_digest, frozen_project_digest)?;
    let sequence = SequenceIdentity::from_authored(study, &workload.model.sequence)?;
    let step = StepIdentity::from_authored(sequence, step_contract)?;
    let constraints = ConstraintPlanIdentity::new(
        step,
        mesh_digest,
        contract_digest("constraints", &workload.model.constraints)?,
    )?;
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
                Vec::new(),
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
        contract_digest("model", &workload.model)?,
        None,
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
    let accuracy =
        AccuracyIdentity::new(step, contract_digest("accuracy", &step_contract.accuracy)?)?;
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
        accuracy,
    )?;
    let execution_plan = ExecutionPlanIdentity::new(
        numerical,
        contract_digest(
            "execution",
            &(
                ANALYSIS_HOST_ABI,
                workload.require_solver_mesh_for_solid,
                AnalysisNumericalObjectKind::SemanticState,
            ),
        )?,
    )?;
    Ok(StructuralIdentities {
        study,
        step,
        problem,
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
    state: &PreparedAnalysisObjects,
    result: &StructuralLinearWorkloadResult,
) -> AnalysisBridgeResult<()> {
    let workload = StructuralLinearWorkload::from_canonical_bytes(&state.payload())?;
    let step = workload.model.sequence.steps.first().ok_or_else(|| {
        AnalysisBridgeError::Invalid("structural workload has no procedure step".into())
    })?;
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
