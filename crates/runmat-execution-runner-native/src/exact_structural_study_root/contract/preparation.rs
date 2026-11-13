//! Freeze whole-study root requests and original materialization inventory.
use super::*;

pub fn prepare_exact_structural_study_root(
    preparation: ExactStructuralStudyRootPreparation,
) -> NativeExecutionResult<PreparedExactStructuralStudyRoot> {
    let ExactStructuralStudyRootPreparation {
        program_revision,
        artifact_access,
        authored_study_digest,
        frozen_project_digest,
        geometry_source_digest,
        workloads,
        local_execution,
        analysis_accelerator_requests,
        input_materializations,
        exact_meshing,
    } = preparation;
    artifact_access.validate().map_err(protocol)?;
    let input_materializations = input_materializations
        .into_iter()
        .map(|materialization| materialization.rebind(&artifact_access).map_err(protocol))
        .collect::<NativeExecutionResult<Vec<_>>>()?;
    let exact_request = exact_meshing.request().clone();
    let expected_meshing_request = exact_meshing.meshing_request().clone();
    let exact_programs = exact_meshing.required_programs()?;
    validate_input_materialization_inventory(
        &workloads,
        &input_materializations,
        &program_revision,
    )?;
    let mut input_value_objects = BTreeMap::new();
    for materialization in &input_materializations {
        materialization.validate_closed().map_err(protocol)?;
        for object in materialization.input_objects() {
            match input_value_objects.get(&object.reference.id) {
                Some(existing) if existing != object => {
                    return Err(protocol(
                        "signal input value id collides with different immutable content",
                    ))
                }
                Some(_) => {}
                None => {
                    input_value_objects.insert(object.reference.id, object.clone());
                }
            }
        }
    }
    let input_expectations = input_materializations
        .iter()
        .map(InputMaterializationPlan::expectation)
        .collect::<Result<Vec<_>, _>>()
        .map_err(protocol)?;
    if exact_request.recipe.program_revision != program_revision {
        return Err(protocol("study and exact-meshing program revisions differ"));
    }
    for reference in object_references(&exact_request.arguments)?.values() {
        if reference.authorization_scope != artifact_access.authorization_scope
            || reference.encryption_context != artifact_access.encryption_context
        {
            return Err(protocol(
                "study and exact-meshing artifact authorities differ",
            ));
        }
    }
    let root = ExactStructuralStudyRootRequest {
        schema_version: 4,
        authored_study_digest,
        frozen_project_digest,
        geometry_source_digest,
        workloads,
        local_execution,
        analysis_accelerator_requests,
        input_materializations: input_materializations
            .iter()
            .map(InputMaterializationPlanRequest::from_plan)
            .collect::<NativeExecutionResult<Vec<_>>>()?,
        exact_meshing: descriptor(&exact_request),
        exact_argument_count: u32::try_from(exact_request.arguments.len()).map_err(protocol)?,
        materialization_argument_count: u32::try_from(input_value_objects.len())
            .map_err(protocol)?,
    };
    let root_object = LogicalObject::new(
        ObjectNamespace::InputValue,
        "analysis/exact-structural-study-roots",
        ROOT_MEDIA_TYPE,
        root.canonical_bytes()?,
    )
    .map_err(protocol)?;
    let root_reference = artifact_access
        .reference(&root_object, ValueRefKind::DriverObject, ROOT_VALUE_SCHEMA)
        .map_err(protocol)?;
    let recipe = ProgramBuildRecipe {
        schema_version: PROGRAM_BUILD_RECIPE_SCHEMA_VERSION,
        program_revision,
        entrypoint: "analysis_workload".into(),
        outputs: OutputContract {
            requested_outputs: 1,
            owned_work: None,
        },
        execution_mode: EXACT_STRUCTURAL_STUDY_ROOT_EXECUTION_MODE.into(),
        target: ProgramTarget::portable(EXACT_STRUCTURAL_STUDY_ROOT_TARGET_PROFILE),
        interop: runmat_types::InteropManifest::empty(),
        accelerators: Vec::new(),
        features: Default::default(),
        compile_options: Default::default(),
        source_objects: Vec::new(),
        foreign_policy: None,
        expected_artifact_id: None,
    };
    let artifact = ProgramArtifact::materialize(
        &recipe,
        ExecutableForm::AnalysisWorkload,
        ROOT_PROGRAM_BYTES.to_vec(),
    )
    .map_err(protocol)?;
    let mut arguments = vec![ValuePayload::Object(Box::new(root_reference))];
    arguments.extend(exact_request.arguments);
    arguments.extend(
        input_value_objects
            .values()
            .map(|object| ValuePayload::Object(Box::new(object.reference.clone()))),
    );
    let request = ProgramExecutionRequest {
        console: None,
        environment: runmat_execution::EnvironmentSnapshot::default(),
        schema_version: PROGRAM_EXECUTION_REQUEST_SCHEMA_VERSION,
        recipe,
        artifact,
        callable: runmat_execution::ProgramCallable::semantic(
            runmat_execution::ProgramFunctionId(0),
            None,
        ),
        context: runmat_execution::ProgramInvocationContext::Direct,
        assignment: None,
        job_id: None,
        arguments,
        requested_outputs: 1,
    };
    request.validate().map_err(protocol)?;
    let mut objects = exact_meshing.objects().to_vec();
    objects.push(root_object);
    let mut required_programs = vec![RequiredStudyProgram {
        recipe: request.recipe.clone(),
        artifact: request.artifact.clone(),
    }];
    for MeshingHostProgram { recipe, artifact } in exact_programs {
        required_programs.push(RequiredStudyProgram { recipe, artifact });
    }
    for materialization in &input_materializations {
        let descriptor = materialization.descriptor().map_err(protocol)?;
        required_programs.push(RequiredStudyProgram {
            recipe: descriptor.recipe,
            artifact: descriptor.artifact,
        });
    }
    let AnalysisHostProgramArtifact { recipe, artifact } =
        prepare_analysis_host_program(request.recipe.program_revision.clone()).map_err(protocol)?;
    required_programs.push(RequiredStudyProgram { recipe, artifact });
    let mut seen = BTreeSet::new();
    required_programs.retain(|program| seen.insert(program.artifact.id));
    Ok(PreparedExactStructuralStudyRoot {
        request,
        objects,
        input_value_objects: input_value_objects.into_values().collect(),
        root,
        artifact_access,
        required_programs,
        expected_meshing_request,
        input_expectations,
    })
}
