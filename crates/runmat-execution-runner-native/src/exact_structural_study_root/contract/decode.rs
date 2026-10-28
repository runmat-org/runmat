//! Authenticate and decode original whole-study root requests.
use super::*;

pub fn decode_exact_structural_study_root(
    program: &ProgramExecutionRequest,
    store: &impl CacheImport,
) -> NativeExecutionResult<DecodedExactStructuralStudyRoot> {
    program.validate_for_portable_host().map_err(protocol)?;
    if program.artifact.form != ExecutableForm::AnalysisWorkload
        || program.recipe.execution_mode != EXACT_STRUCTURAL_STUDY_ROOT_EXECUTION_MODE
        || program.recipe.target
            != ProgramTarget::portable(EXACT_STRUCTURAL_STUDY_ROOT_TARGET_PROFILE)
        || program.artifact.executable_bytes != ROOT_PROGRAM_BYTES
        || !program.recipe.compile_options.is_empty()
    {
        return Err(protocol("program is not an exact structural study root"));
    }
    let [ValuePayload::Object(root_reference), component_arguments @ ..] =
        program.arguments.as_slice()
    else {
        return Err(protocol("exact structural study root inputs are invalid"));
    };
    let artifact_access = ArtifactAccess {
        authorization_scope: root_reference.authorization_scope.clone(),
        encryption_context: root_reference.encryption_context,
    };
    artifact_access
        .validate_reference(
            root_reference,
            ValueRefKind::DriverObject,
            ROOT_MEDIA_TYPE,
            ROOT_VALUE_SCHEMA,
        )
        .map_err(protocol)?;
    let bytes = read(store, root_reference)?;
    let root = ExactStructuralStudyRootRequest::decode(&bytes)?;
    let exact_argument_count = usize::try_from(root.exact_argument_count).map_err(protocol)?;
    let materialization_argument_count =
        usize::try_from(root.materialization_argument_count).map_err(protocol)?;
    let component_argument_count = exact_argument_count
        .checked_add(materialization_argument_count)
        .ok_or_else(|| protocol("exact structural study root argument count overflows"))?;
    if component_arguments.len() != component_argument_count
        || root.exact_meshing.recipe.program_revision != program.recipe.program_revision
    {
        return Err(protocol(
            "exact structural study root component contract differs from its request",
        ));
    }
    let (exact_arguments, materialization_arguments) =
        component_arguments.split_at(exact_argument_count);
    let exact_meshing = ProgramExecutionRequest {
        console: root.exact_meshing.console,
        environment: root.exact_meshing.environment.clone(),
        schema_version: PROGRAM_EXECUTION_REQUEST_SCHEMA_VERSION,
        recipe: root.exact_meshing.recipe.clone(),
        artifact: root.exact_meshing.artifact.clone(),
        callable: root.exact_meshing.callable.clone(),
        context: runmat_execution::ProgramInvocationContext::Direct,
        assignment: None,
        job_id: None,
        arguments: exact_arguments.to_vec(),
        requested_outputs: root.exact_meshing.requested_outputs,
    };
    let decoded_exact = decode_exact_meshing_root(&exact_meshing, store)?;
    if decoded_exact.artifact_access != artifact_access
        || Digest::sha256(&decoded_exact.source_bytes) != root.geometry_source_digest
    {
        return Err(protocol(
            "exact structural study geometry differs from its semantic intent",
        ));
    }
    let input_materializations = root
        .input_materializations
        .iter()
        .map(InputMaterializationPlanRequest::to_plan)
        .collect::<NativeExecutionResult<Vec<_>>>()?;
    let expected_materialization_references = input_materializations
        .iter()
        .flat_map(|prepared| prepared.input_references().iter())
        .map(|reference| (reference.id, reference))
        .collect::<BTreeMap<_, _>>();
    let transported_materialization_references = materialization_arguments
        .iter()
        .map(|argument| match argument {
            ValuePayload::Object(reference) => {
                artifact_access
                    .validate_reference(
                        reference,
                        ValueRefKind::DriverObject,
                        runmat_execution::value::EXTERNALIZED_VALUE_MEDIA_TYPE,
                        runmat_execution::value::EXTERNALIZED_VALUE_SCHEMA,
                    )
                    .map_err(protocol)?;
                Ok((reference.id, reference.as_ref()))
            }
            _ => Err(protocol("signal input value is not externalized")),
        })
        .collect::<NativeExecutionResult<BTreeMap<_, _>>>()?;
    if expected_materialization_references != transported_materialization_references {
        return Err(protocol(
            "exact structural study signal references differ from its materialization plans",
        ));
    }
    Ok(DecodedExactStructuralStudyRoot {
        authored_study_digest: root.authored_study_digest,
        frozen_project_digest: root.frozen_project_digest,
        geometry_source_digest: root.geometry_source_digest,
        workloads: root.workloads,
        local_execution: root.local_execution,
        memory_per_worker_bytes: None,
        analysis_accelerator_requests: root.analysis_accelerator_requests,
        input_materializations,
        exact_meshing,
        meshing_request: decoded_exact.request.meshing_request,
        artifact_access,
        compiled_study: None,
        compiled_geometry_output: None,
    })
}
