//! Portable, assembly-free admission for the transient checkpoint-scope producer.

use runmat_analysis_fea::{
    execution::{
        AnalysisNumericalObjectKind, AnalysisWorkKind, AnalysisWorkloadIdentity,
        CanonicalSemanticState, ConstraintPlanIdentity, DiscretizationIdentity,
        ExecutionPlanIdentity, NumericalPlanIdentity, OperatorIdentity, ProblemIdentity,
        SequenceIdentity, StepIdentity, StructuralTransientCheckpointScope,
        StructuralTransientCoordinateSpace, StructuralTransientSolverPlan,
        StructuralTransientWorkload, StudyIdentity,
    },
    planning::resource::{FeaResourceDemand, FeaResourcePhase, FeaResourcePlan},
};
use runmat_canonical_codec::CanonicalLimits;
use runmat_execution::{Digest, ProgramRevision};
use runmat_execution_artifact::{
    cache::CacheImport, object::ObjectInventoryLimits, ArtifactAccess, ObjectNamespace,
    ProgramExecutionResponse,
};
use runmat_meshing_core::{CanonicalMeshingContract, SolverMeshArtifact};
use serde::Serialize;

use crate::{
    decode_prepared_semantic_state, decode_prepared_structural_transient_scope,
    decode_prepared_structural_transient_workload, import_analysis_objects,
    import_analysis_output_set, prepare_analysis_objects, prepare_analysis_program,
    prepare_semantic_state_objects, prepared_input_resource_facts,
    projected_output_set_resource_facts, AnalysisBridgeError, AnalysisBridgeResult,
    AnalysisHostWorkload, PreparedAnalysisObjects, PreparedAnalysisProgram,
};

const TASK_PREFIX: &[u8] = b"runmat-analysis-transient-scope-task/v1\0";
/// Identity digests hash the authored model and constraint plan, which grow with generated
/// signals and the mesh; only the digest is retained, never the encoded bytes.
const IDENTITY_LIMITS: CanonicalLimits = CanonicalLimits::new(0, 0, 65_536, 64);

pub struct StructuralTransientScopeTaskPreparation {
    pub local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan,
    pub workload: StructuralTransientWorkload,
    pub mesh: SolverMeshArtifact,
    pub modal_state: Option<CanonicalSemanticState>,
    pub artifact_access: ArtifactAccess,
    pub program_revision: ProgramRevision,
    pub chunk_bytes: usize,
    pub inventory_limits: ObjectInventoryLimits,
}

pub struct PreparedStructuralTransientScopeTask {
    pub host: AnalysisHostWorkload,
    pub program: PreparedAnalysisProgram,
    pub inputs: Vec<PreparedAnalysisObjects>,
    pub resource_plan: FeaResourcePlan,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ImportedStructuralTransientScope {
    pub scope: StructuralTransientCheckpointScope,
    pub objects: PreparedAnalysisObjects,
    pub response: ProgramExecutionResponse,
}

pub fn prepare_structural_transient_scope_task(
    preparation: StructuralTransientScopeTaskPreparation,
) -> AnalysisBridgeResult<PreparedStructuralTransientScopeTask> {
    preparation.workload.validate()?;
    preparation
        .mesh
        .validate_canonical()
        .map_err(|error| invalid(error.to_string()))?;
    if preparation.chunk_bytes == 0 {
        return Err(invalid(
            "transient scope task requires a nonzero chunk size",
        ));
    }
    let workload_bytes = preparation.workload.canonical_bytes()?;
    let workload_objects = prepare_analysis_objects(
        AnalysisNumericalObjectKind::StructuralTransientWorkload,
        Digest::sha256(&workload_bytes),
        &workload_bytes,
        preparation.chunk_bytes,
        ObjectNamespace::InputValue,
        preparation.inventory_limits,
    )?;
    let mesh_objects = prepare_analysis_objects(
        AnalysisNumericalObjectKind::Mesh,
        Digest::from_bytes(*preparation.mesh.canonical_digest.bytes()),
        &preparation
            .mesh
            .canonical_encode()
            .map_err(|error| invalid(error.to_string()))?,
        preparation.chunk_bytes,
        ObjectNamespace::InputValue,
        preparation.inventory_limits,
    )?;
    let mut numerical_inputs = vec![workload_objects, mesh_objects];
    if let Some(state) = &preparation.modal_state {
        numerical_inputs.push(prepare_semantic_state_objects(
            state,
            preparation.chunk_bytes,
            ObjectNamespace::ResultValue,
            preparation.inventory_limits,
        )?);
    }
    let resource_plan =
        runmat_analysis_fea::pipeline::transient::plan_transient_checkpoint_scope_task(
            &preparation.workload,
            &preparation.mesh,
            preparation.modal_state.as_ref(),
            prepared_input_resource_facts(&numerical_inputs)?,
            projected_output_set_resource_facts(),
            preparation.local_execution,
        )?;
    let (study, step, execution_plan) =
        transient_scope_task_identity(&preparation.workload, &preparation.mesh, &resource_plan)?;
    let resource_objects = prepare_analysis_objects(
        AnalysisNumericalObjectKind::ResourcePlan,
        resource_plan.identity.digest(),
        &resource_plan
            .canonical_bytes()
            .map_err(|error| invalid(error.to_string()))?,
        preparation.chunk_bytes,
        ObjectNamespace::InputValue,
        preparation.inventory_limits,
    )?;
    let mut inputs = numerical_inputs;
    inputs.push(resource_objects);
    let host = crate::portable_task_support::task_host(
        crate::portable_task_support::PortableTaskHostPreparation {
            study,
            step,
            execution_plan,
            work_kind: AnalysisWorkKind::PrepareStructuralTransientCheckpointScope,
            inputs: &inputs,
            outputs: vec![AnalysisNumericalObjectKind::StructuralTransientCheckpointScope],
            artifact_access: preparation.artifact_access,
            output_policy: crate::AnalysisOutputPolicy::new(
                preparation.chunk_bytes,
                preparation.inventory_limits,
            )?,
        },
    )?;
    let program = prepare_analysis_program(&host, preparation.program_revision, &inputs)?;
    Ok(PreparedStructuralTransientScopeTask {
        host,
        program,
        inputs,
        resource_plan,
    })
}

/// The operator identity here names the symbolic operator *scope* admitted for construction,
/// not the numerical K/M/C identities that only the worker may discover after assembly.
pub fn transient_scope_task_identity(
    workload: &StructuralTransientWorkload,
    mesh: &SolverMeshArtifact,
    resource_plan: &FeaResourcePlan,
) -> AnalysisBridgeResult<(StudyIdentity, StepIdentity, ExecutionPlanIdentity)> {
    workload.validate()?;
    mesh.validate_canonical()
        .map_err(|error| invalid(error.to_string()))?;
    resource_plan
        .validate_identity()
        .map_err(|error| invalid(error.to_string()))?;
    let FeaResourceDemand::StructuralTransientScopePreparation(demand) =
        &resource_plan.input.demand
    else {
        return Err(invalid(
            "transient scope task has the wrong resource demand",
        ));
    };
    let workload_digest = Digest::sha256(workload.canonical_bytes()?);
    let mesh_digest = Digest::from_bytes(*mesh.canonical_digest.bytes());
    if resource_plan.input.phase
        != FeaResourcePhase::StructuralTransientScopePreparation(workload_digest)
        || demand.source_workload_digest != workload_digest
        || demand.operator.topology_digest != mesh_digest
    {
        return Err(AnalysisBridgeError::Identity(
            "transient scope resource plan differs from its workload or mesh",
        ));
    }
    let study = StudyIdentity::from_authored_contract(&workload.model, None)?;
    let sequence = SequenceIdentity::from_solver_contract(&workload.model.sequence)?;
    let step = StepIdentity::from_solver_contract(sequence, workload.procedure_step()?)?;
    let constraints = ConstraintPlanIdentity::new(
        step,
        mesh_digest,
        digest("scope-constraints", &demand.operator.constraints)?,
    )?;
    let problem = ProblemIdentity::new(
        step,
        digest("scope-model", &workload.model.solver_identity_contract())?,
        None,
        constraints,
        Vec::new(),
    )?;
    let discretization = DiscretizationIdentity::new(
        problem,
        mesh_digest,
        digest("scope-formulation", &mesh_digest)?,
    )?;
    let operator = OperatorIdentity::new(discretization, demand.operator.numerical_scope_digest)?;
    let numerical = NumericalPlanIdentity::new(
        operator,
        digest(
            "scope-solver",
            &(
                &workload.solver,
                workload.time_step_policy,
                workload.residual_target,
            ),
        )?,
    )?;
    let execution = ExecutionPlanIdentity::new(
        numerical,
        digest("scope-preparation", &resource_plan.identity)?,
    )?;
    Ok((study, step, execution))
}

pub fn import_structural_transient_scope(
    source: &impl CacheImport,
    response: ProgramExecutionResponse,
    prepared: &PreparedStructuralTransientScopeTask,
    limits: ObjectInventoryLimits,
) -> AnalysisBridgeResult<ImportedStructuralTransientScope> {
    response.validate_against(&prepared.program.request)?;
    let retained_response = response.clone();
    let output_set = import_analysis_output_set(source, response, &prepared.host)?;
    let [output] = output_set.manifest.outputs.as_slice() else {
        return Err(invalid(
            "transient scope task must return exactly one scope",
        ));
    };
    let objects = import_analysis_objects(
        source,
        &output.root,
        AnalysisNumericalObjectKind::StructuralTransientCheckpointScope,
        output.logical_identity,
        output.payload_digest,
        &prepared.host.artifact_access,
        limits,
    )?;
    let scope = decode_prepared_structural_transient_scope(&objects)?;
    let [workload_input, mesh_input, remaining @ ..] = prepared.inputs.as_slice() else {
        return Err(invalid(
            "transient scope task omits its prepared numerical inputs",
        ));
    };
    let workload = decode_prepared_structural_transient_workload(workload_input)?;
    let mesh = SolverMeshArtifact::canonical_decode(&mesh_input.payload())
        .map_err(|error| invalid(error.to_string()))?;
    let modal_state = match remaining {
        [_resource] => None,
        [state, _resource] => Some(decode_prepared_semantic_state(state)?),
        _ => return Err(invalid("transient scope task has an invalid input shape")),
    };
    let mesh_digest = Digest::from_bytes(*mesh.canonical_digest.bytes());
    let mut dependencies = vec![mesh_digest];
    let coordinate_space_matches = match (&workload.solver, &scope.coordinate_space, &modal_state) {
        (
            StructuralTransientSolverPlan::Direct,
            StructuralTransientCoordinateSpace::Direct,
            None,
        ) => true,
        (
            StructuralTransientSolverPlan::ModalReduced { basis_transition },
            StructuralTransientCoordinateSpace::ModalReduced {
                basis_state,
                basis,
                coordinate_count,
                ..
            },
            Some(state),
        ) => {
            dependencies.extend([
                basis_transition.source_state.digest(),
                basis_transition.identity.digest(),
            ]);
            state.identity == *basis_state
                && state.identity == basis_transition.source_state
                && state.topology_digest == mesh_digest
                && state.payload.modal_basis().is_some_and(|value| {
                    value.identity() == *basis && value.modes().len() == *coordinate_count
                })
        }
        _ => false,
    };
    let expected_source = AnalysisWorkloadIdentity::new(
        scope.study,
        scope.step,
        scope.execution_plan,
        AnalysisWorkKind::SolveStructuralTransientBlock,
        Digest::sha256(workload.canonical_bytes()?),
        dependencies,
    )?;
    if scope.study != prepared.host.workload.study
        || scope.step != prepared.host.workload.step
        || scope.source_workload != expected_source
        || !coordinate_space_matches
    {
        return Err(AnalysisBridgeError::Identity(
            "prepared transient scope differs from its exact workload, mesh, or modal basis",
        ));
    }
    let mut expected_objects = objects.unique_references(
        &prepared.host.artifact_access,
        runmat_execution::value::ValueRefKind::ResultObject,
    )?;
    expected_objects.push(output_set.root);
    if output_set.result_objects != expected_objects {
        return Err(AnalysisBridgeError::Identity(
            "transient scope result inventory differs from its canonical closure",
        ));
    }
    Ok(ImportedStructuralTransientScope {
        scope,
        objects,
        response: retained_response,
    })
}

fn digest<T: Serialize>(domain: &'static str, input: &T) -> AnalysisBridgeResult<Digest> {
    Ok(
        runmat_analysis_fea::contracts::canonical_digest::canonical_contract_digest(
            TASK_PREFIX,
            domain,
            input,
            IDENTITY_LIMITS,
        )?,
    )
}

fn invalid(message: impl Into<String>) -> AnalysisBridgeError {
    AnalysisBridgeError::Invalid(message.into())
}
