use std::collections::BTreeSet;

use runmat_analysis_fea::execution::{
    AnalysisNumericalObjectKind, AnalysisWorkKind, CanonicalSemanticState,
    StructuralTransientBlockRequest, StructuralTransientCheckpoint,
    StructuralTransientCoordinateSpace, StructuralTransientSegment, StructuralTransientSolverPlan,
    StructuralTransientWorkload,
};
use runmat_analysis_fea::planning::resource::{
    build_structural_transient_block_resource_plan, FeaResourcePlan,
};
use runmat_execution::value::ValueRefKind;
use runmat_execution::ProgramRevision;
use runmat_execution_artifact::cache::CacheImport;
use runmat_execution_artifact::object::ObjectInventoryLimits;
use runmat_execution_artifact::{ObjectNamespace, ProgramExecutionResponse};
use runmat_meshing_core::CanonicalMeshingContract;

use crate::{
    decode_prepared_structural_transient_checkpoint, decode_prepared_structural_transient_segment,
    decode_prepared_structural_transient_workload, import_analysis_objects,
    import_analysis_output_set, prepare_analysis_objects, prepare_analysis_program,
    prepared_input_resource_facts, projected_output_set_resource_facts, AnalysisBridgeError,
    AnalysisBridgeResult, AnalysisCapabilityRequirement, AnalysisHostWorkload,
    AnalysisWorkloadInput, AnalysisWorkloadRequest, PreparedAnalysisObjects,
    PreparedAnalysisProgram, ANALYSIS_HOST_ABI,
};

pub struct StructuralTransientBlockPreparation {
    pub local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan,
    pub workload: PreparedAnalysisObjects,
    pub mesh: PreparedAnalysisObjects,
    pub request: PreparedAnalysisObjects,
    pub predecessor: Option<PreparedAnalysisObjects>,
    pub modal_state: Option<PreparedAnalysisObjects>,
    pub artifact_access: runmat_execution_artifact::ArtifactAccess,
    pub program_revision: ProgramRevision,
    pub chunk_bytes: usize,
    pub inventory_limits: ObjectInventoryLimits,
}

pub struct PreparedStructuralTransientBlockTask {
    pub workload: StructuralTransientWorkload,
    pub request: StructuralTransientBlockRequest,
    pub predecessor: Option<StructuralTransientCheckpoint>,
    pub modal_state: Option<CanonicalSemanticState>,
    pub host: AnalysisHostWorkload,
    pub program: PreparedAnalysisProgram,
    pub inputs: Vec<PreparedAnalysisObjects>,
    pub resource_plan: FeaResourcePlan,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ImportedStructuralTransientBlock {
    pub segment: StructuralTransientSegment,
    pub objects: PreparedAnalysisObjects,
}

pub fn prepare_structural_transient_block_task(
    preparation: StructuralTransientBlockPreparation,
) -> AnalysisBridgeResult<PreparedStructuralTransientBlockTask> {
    let workload = decode_prepared_structural_transient_workload(&preparation.workload)?;
    if preparation.mesh.manifest.kind != AnalysisNumericalObjectKind::Mesh {
        return Err(AnalysisBridgeError::Invalid(
            "structural transient block requires one canonical mesh object".into(),
        ));
    }
    let request = crate::decode_prepared_structural_transient_block_request(&preparation.request)?;
    let predecessor = preparation
        .predecessor
        .as_ref()
        .map(decode_prepared_structural_transient_checkpoint)
        .transpose()?;
    let modal_state = preparation
        .modal_state
        .as_ref()
        .map(crate::decode_prepared_semantic_state)
        .transpose()?;
    request.validate_against_predecessor(predecessor.as_ref())?;
    let solver_mesh =
        runmat_meshing_core::SolverMeshArtifact::canonical_decode(&preparation.mesh.payload())
            .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    if preparation.workload.manifest.logical_identity != request.scope.source_workload.digest()
        || preparation.mesh.manifest.logical_identity
            != runmat_execution::Digest::from_bytes(*solver_mesh.canonical_digest.bytes())
    {
        return Err(AnalysisBridgeError::Identity(
            "structural transient workload or mesh differs from the block scope",
        ));
    }
    let modal_state_matches = match (
        &workload.solver,
        &request.scope.coordinate_space,
        modal_state.as_ref(),
    ) {
        (
            StructuralTransientSolverPlan::Direct,
            StructuralTransientCoordinateSpace::Direct,
            None,
        ) => true,
        (
            StructuralTransientSolverPlan::ModalReduced { basis_transition },
            StructuralTransientCoordinateSpace::ModalReduced {
                basis_state: scoped_basis,
                ..
            },
            Some(state),
        ) => {
            state.identity == basis_transition.source_state
                && state.identity == *scoped_basis
                && state.topology_digest
                    == runmat_execution::Digest::from_bytes(*solver_mesh.canonical_digest.bytes())
        }
        _ => false,
    };
    if !modal_state_matches {
        return Err(AnalysisBridgeError::Identity(
            "structural transient modal state differs from its workload and checkpoint scope",
        ));
    }
    let mut inputs = vec![preparation.workload, preparation.mesh, preparation.request];
    let mut reference_kinds = vec![ValueRefKind::DriverObject; inputs.len()];
    if let Some(predecessor) = preparation.predecessor {
        inputs.push(predecessor);
        reference_kinds.push(ValueRefKind::ResultObject);
    }
    if let Some(modal_state) = preparation.modal_state {
        inputs.push(modal_state);
        reference_kinds.push(ValueRefKind::ResultObject);
    }
    let resource_plan = build_structural_transient_block_resource_plan(
        &workload,
        &solver_mesh,
        modal_state.as_ref(),
        &request,
        predecessor.as_ref(),
        Some(prepared_input_resource_facts(&inputs)?),
        Some(projected_output_set_resource_facts()),
        preparation.local_execution,
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
    inputs.push(resource_objects);
    reference_kinds.push(ValueRefKind::DriverObject);
    let workload_inputs = inputs
        .iter()
        .zip(reference_kinds)
        .map(|(input, reference_kind)| AnalysisWorkloadInput {
            kind: input.manifest.kind,
            logical_identity: input.manifest.logical_identity,
            payload_digest: input.manifest.payload_digest,
            manifest_digest: input.root.digest,
            manifest_encoded_length: input.root.encoded_length,
            reference_kind,
        })
        .collect();
    let analysis_workload = AnalysisWorkloadRequest::new(
        request.scope.study,
        request.scope.step,
        request.scope.execution_plan,
        AnalysisWorkKind::SolveStructuralTransientBlock,
        workload_inputs,
        vec![AnalysisNumericalObjectKind::StructuralTransientSegment],
        BTreeSet::from([AnalysisCapabilityRequirement::Host {
            abi: ANALYSIS_HOST_ABI.into(),
        }]),
    )?;
    let host = AnalysisHostWorkload::new(
        analysis_workload,
        preparation.artifact_access,
        crate::AnalysisOutputPolicy::new(preparation.chunk_bytes, preparation.inventory_limits)?,
    )?;
    let program = prepare_analysis_program(&host, preparation.program_revision, &inputs)?;
    Ok(PreparedStructuralTransientBlockTask {
        workload,
        request,
        predecessor,
        modal_state,
        host,
        program,
        inputs,
        resource_plan,
    })
}

pub fn import_structural_transient_block(
    source: &impl CacheImport,
    response: ProgramExecutionResponse,
    prepared: &PreparedStructuralTransientBlockTask,
    limits: ObjectInventoryLimits,
) -> AnalysisBridgeResult<ImportedStructuralTransientBlock> {
    response.validate_against(&prepared.program.request)?;
    let output_set = import_analysis_output_set(source, response, &prepared.host)?;
    let [output] = output_set.manifest.outputs.as_slice() else {
        return Err(AnalysisBridgeError::Invalid(
            "structural transient block must return exactly one segment".into(),
        ));
    };
    let objects = import_analysis_objects(
        source,
        &output.root,
        AnalysisNumericalObjectKind::StructuralTransientSegment,
        output.logical_identity,
        output.payload_digest,
        &prepared.host.artifact_access,
        limits,
    )?;
    let segment = decode_prepared_structural_transient_segment(&objects)?;
    if segment.manifest().request != prepared.request
        || segment.manifest().terminal_checkpoint.predecessor
            != prepared.predecessor.as_ref().map(|value| value.identity)
    {
        return Err(AnalysisBridgeError::Identity(
            "structural transient segment differs from its prepared block lineage",
        ));
    }
    let mut expected_objects =
        objects.unique_references(&prepared.host.artifact_access, ValueRefKind::ResultObject)?;
    expected_objects.push(output_set.root);
    if output_set.result_objects != expected_objects {
        return Err(AnalysisBridgeError::Identity(
            "structural transient block result inventory differs from its canonical closure",
        ));
    }
    Ok(ImportedStructuralTransientBlock { segment, objects })
}
