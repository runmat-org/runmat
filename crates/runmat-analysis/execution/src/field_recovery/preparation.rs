use super::*;
use crate::{
    prepare_analysis_objects, prepare_analysis_program, AnalysisCapabilityRequirement,
    AnalysisWorkloadInput, AnalysisWorkloadRequest, ANALYSIS_HOST_ABI,
};
use runmat_analysis_fea::execution::StructuralLinearWorkload;
use runmat_analysis_fea::results::field::StructuralRecoveredFieldTopology;
use runmat_execution::value::ValueRefKind;
use runmat_meshing_core::{CanonicalMeshingContract, SolverMeshArtifact};
use std::collections::BTreeSet;
pub fn prepare_structural_field_recovery(
    preparation: StructuralFieldRecoveryPreparation,
) -> AnalysisBridgeResult<PreparedStructuralFieldRecovery> {
    preparation.source_host.validate()?;
    if preparation.source_host.workload.kind != AnalysisWorkKind::Solve
        || preparation.source_result.manifest.kind != AnalysisNumericalObjectKind::SolverResult
        || preparation.source_invocation.manifest.kind
            != AnalysisNumericalObjectKind::SolverInvocation
        || preparation
            .mesh
            .as_ref()
            .is_some_and(|mesh| mesh.manifest.kind != AnalysisNumericalObjectKind::Mesh)
        || preparation.chunk_bytes == 0
    {
        return Err(AnalysisBridgeError::Invalid(
            "field recovery requires one completed solve, its invocation, optional canonical mesh, and bounded chunks"
                .into(),
        ));
    }
    let source_invocation = preparation
        .source_host
        .workload
        .inputs
        .iter()
        .find(|input| input.kind == AnalysisNumericalObjectKind::SolverInvocation)
        .ok_or_else(|| {
            AnalysisBridgeError::Invalid("source solve has no invocation input".into())
        })?;
    if source_invocation.logical_identity != preparation.source_invocation.manifest.logical_identity
        || source_invocation.payload_digest != preparation.source_invocation.manifest.payload_digest
        || source_invocation.manifest_digest != preparation.source_invocation.root.digest
    {
        return Err(AnalysisBridgeError::Identity(
            "field recovery invocation differs from the source solve invocation",
        ));
    }
    let source_mesh = preparation
        .source_host
        .workload
        .inputs
        .iter()
        .find(|input| input.kind == AnalysisNumericalObjectKind::Mesh);
    match (source_mesh, preparation.mesh.as_ref()) {
        (Some(source), Some(mesh))
            if source.logical_identity == mesh.manifest.logical_identity
                && source.payload_digest == mesh.manifest.payload_digest
                && source.manifest_digest == mesh.root.digest => {}
        (None, None) => {}
        _ => {
            return Err(AnalysisBridgeError::Identity(
                "field recovery mesh presence or identity differs from the source solve",
            ))
        }
    }
    let (source_result, result) =
        crate::structural::result_source::decode_structural_result_source(
            &preparation.source_host,
            &preparation.source_result,
        )?;
    let workload =
        StructuralLinearWorkload::from_canonical_bytes(&preparation.source_invocation.payload())?;
    result.validate_thermal_workload(&workload)?;
    let mesh_artifact = preparation
        .mesh
        .as_ref()
        .map(|mesh| {
            SolverMeshArtifact::canonical_decode(&mesh.payload())
                .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))
        })
        .transpose()?;
    let topology =
        StructuralRecoveredFieldTopology::derive(&workload.model, mesh_artifact.as_ref())
            .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    if result.constraint_result.topology != topology.identity {
        return Err(AnalysisBridgeError::Identity(
            "field recovery topology differs from the completed solve",
        ));
    }
    let source_facts = crate::object::prepared_input_resource_facts_from_refs(
        [&preparation.source_invocation, &preparation.source_result]
            .into_iter()
            .chain(preparation.mesh.iter()),
    )?;
    let compiler_limit =
        runmat_analysis_fea::planning::resource::constraint_compilation_available_bytes(
            workload
                .procedure_step()?
                .accuracy
                .resources
                .resident_byte_budget(),
            &source_facts,
            runmat_analysis_fea::planning::resource::ByteCount(0),
        )?;
    let compilation =
        runmat_analysis_fea::constraints::CanonicalStructuralConstraintPlan::compile_for_model_with_workspace_limit(
            &workload.model,
            mesh_artifact.as_ref(), compiler_limit,
        )
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let constraint_plan = compilation.plan();
    result
        .constraint_result
        .validate_against_model(constraint_plan, &workload.model, mesh_artifact.as_ref())
        .map_err(|_| {
            AnalysisBridgeError::Identity(
                "field recovery constraint result differs from the canonical executed plan",
            )
        })?;
    let connector_result =
        result
            .run
            .structural_connector_result
            .as_deref()
            .ok_or(AnalysisBridgeError::Identity(
                "field recovery source has no canonical connector result",
            ))?;
    if connector_result.plan.topology != topology.identity {
        return Err(AnalysisBridgeError::Identity(
            "field recovery connector topology differs from the completed solve",
        ));
    }
    connector_result
        .plan
        .validate_against_authored(&workload.model.connectors)
        .map_err(|_| {
            AnalysisBridgeError::Identity(
                "field recovery connector result differs from the authored connector contract",
            )
        })?;
    let recovery_plan = runmat_analysis_fea::execution::FieldRecoveryPlan::resolve_structural(
        workload.procedure_step()?,
        topology,
        constraint_plan.groups.len(),
    )?;
    let request = FieldRecoveryRequest::new(
        preparation.source_host.workload.step,
        FieldRecoverySource::SolverResult(source_result),
        runmat_analysis_fea::execution::FieldTopologyIdentity::Structural {
            topology: topology.identity,
        },
        recovery_plan,
    )?;
    let request_bytes = request.canonical_bytes()?;
    let request_objects = prepare_analysis_objects(
        AnalysisNumericalObjectKind::FieldRecoveryRequest,
        request.identity().digest(),
        &request_bytes,
        preparation.chunk_bytes,
        ObjectNamespace::InputValue,
        preparation.inventory_limits,
    )?;
    let inputs = vec![
        (request_objects, ValueRefKind::DriverObject),
        (preparation.source_invocation, ValueRefKind::DriverObject),
        (preparation.source_result, ValueRefKind::ResultObject),
    ];
    let mut inputs = inputs;
    if let Some(mesh) = preparation.mesh {
        inputs.push((mesh, ValueRefKind::DriverObject));
    }
    let policy =
        crate::AnalysisOutputPolicy::new(preparation.chunk_bytes, preparation.inventory_limits)?;
    let resource_plan = resource::build_from_refs(
        &request,
        &workload,
        mesh_artifact.as_ref(),
        &result,
        &compilation,
        inputs.iter().map(|(i, _)| i),
        policy,
        preparation.local_execution,
    )?;
    inputs.push((
        prepare_analysis_objects(
            AnalysisNumericalObjectKind::ResourcePlan,
            resource_plan.identity.digest(),
            &resource_plan.canonical_bytes()?,
            preparation.chunk_bytes,
            ObjectNamespace::InputValue,
            preparation.inventory_limits,
        )?,
        ValueRefKind::DriverObject,
    ));
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
    let workload = AnalysisWorkloadRequest::new(
        preparation.source_host.workload.study,
        preparation.source_host.workload.step,
        preparation.source_host.workload.execution_plan,
        AnalysisWorkKind::RecoverField,
        workload_inputs,
        vec![AnalysisNumericalObjectKind::Field],
        BTreeSet::from([AnalysisCapabilityRequirement::Host {
            abi: ANALYSIS_HOST_ABI.into(),
        }]),
    )?;
    let host = AnalysisHostWorkload::new(
        workload,
        preparation.source_host.artifact_access,
        crate::AnalysisOutputPolicy::new(preparation.chunk_bytes, preparation.inventory_limits)?,
    )?;
    let program = prepare_analysis_program(&host, preparation.program_revision, &inputs)?;
    Ok(PreparedStructuralFieldRecovery {
        request,
        host,
        program,
        inputs,
        resource_plan,
        local_execution: preparation.local_execution,
    })
}
