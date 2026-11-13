//! Construct the closed portable root and its transport inventory.
use super::*;

pub fn prepare_structural_study_root(
    preparation: StructuralStudyRootPreparation,
) -> NativeExecutionResult<PreparedStructuralStudyRoot> {
    let StructuralStudyRootPreparation {
        program_revision,
        artifact_access,
        authored_study_digest,
        frozen_project_digest,
        study,
        materials,
        planning,
        local_execution,
        memory_per_worker_bytes,
        accelerator_admission,
        analysis_accelerator_requests,
        input_materializations,
        initial_fields,
        geometry_meshing,
    } = preparation;
    artifact_access.validate().map_err(protocol)?;
    if geometry_meshing.request().recipe.program_revision != program_revision {
        return Err(protocol(
            "geometry-meshing and structural-study program revisions differ",
        ));
    }
    let expected_meshing_request = geometry_meshing.meshing_request().clone();
    let geometry_request = geometry_meshing.geometry_request().clone();
    let input_materializations = input_materializations
        .into_iter()
        .map(|materialization| materialization.rebind(&artifact_access).map_err(protocol))
        .collect::<NativeExecutionResult<Vec<_>>>()?;
    validate_input_materialization_plan_inventory(
        &study,
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
                        "generator input value id collides with different immutable content",
                    ));
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
    let sources = initial_fields::prepare(&study, &initial_fields, &artifact_access)?;
    let root = StructuralStudyRootRequest {
        schema_version: 4,
        authored_study_digest,
        frozen_project_digest,
        study,
        materials: materials
            .into_iter()
            .map(|(reference, material)| StructuralStudyMaterial {
                reference,
                material,
            })
            .collect(),
        planning,
        local_execution,
        memory_per_worker_bytes,
        accelerator_admission,
        analysis_accelerator_requests,
        input_materializations: input_materializations
            .iter()
            .map(InputMaterializationPlanRequest::from_plan)
            .collect::<NativeExecutionResult<Vec<_>>>()?,
        geometry_meshing: descriptor_from_request(geometry_meshing.request()),
        geometry_argument_count: u32::try_from(geometry_meshing.request().arguments.len())
            .map_err(protocol)?,
        materialization_argument_count: u32::try_from(input_value_objects.len())
            .map_err(protocol)?,
        initial_field_argument_count: u32::try_from(sources.arguments.len()).map_err(protocol)?,
        initial_fields: sources.inputs,
    };
    let root_object = LogicalObject::new(
        ObjectNamespace::InputValue,
        "analysis/structural-study-roots",
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
        execution_mode: STRUCTURAL_STUDY_ROOT_EXECUTION_MODE.into(),
        target: ProgramTarget::portable(STRUCTURAL_STUDY_ROOT_TARGET_PROFILE),
        interop: runmat_types::InteropManifest::empty(),
        accelerators: Vec::new(),
        features: BTreeSet::new(),
        compile_options: BTreeSet::new(),
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
    arguments.extend(geometry_meshing.request().arguments.clone());
    arguments.extend(
        input_value_objects
            .values()
            .map(|object| ValuePayload::Object(Box::new(object.reference.clone()))),
    );
    arguments.extend(sources.arguments);
    let request = ProgramExecutionRequest {
        console: None,
        environment: runmat_execution::EnvironmentSnapshot::default(),
        schema_version: PROGRAM_EXECUTION_REQUEST_SCHEMA_VERSION,
        recipe: recipe.clone(),
        artifact: artifact.clone(),
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
    request.validate_for_portable_host().map_err(protocol)?;
    let mut objects = geometry_meshing.objects().to_vec();
    objects.extend(sources.objects);
    objects.push(root_object);
    let mut object_digests = BTreeSet::new();
    objects.retain(|object| object_digests.insert(object.descriptor.digest));
    let mut required_programs = vec![RequiredStructuralStudyProgram { recipe, artifact }];
    required_programs.extend(
        geometry_meshing
            .required_programs()
            .iter()
            .map(required_program),
    );
    for materialization in &input_materializations {
        let descriptor = materialization.descriptor().map_err(protocol)?;
        required_programs.push(RequiredStructuralStudyProgram {
            recipe: descriptor.recipe,
            artifact: descriptor.artifact,
        });
    }
    let analysis_host =
        prepare_analysis_host_program(request.recipe.program_revision.clone()).map_err(protocol)?;
    required_programs.push(RequiredStructuralStudyProgram {
        recipe: analysis_host.recipe,
        artifact: analysis_host.artifact,
    });
    let mut seen = BTreeSet::new();
    required_programs.retain(|program| seen.insert(program.artifact.id));
    let expectation = StructuralStudyResultExpectation {
        program_revision: request.recipe.program_revision.clone(),
        artifact_access,
        authored_study_digest: root.authored_study_digest,
        frozen_project_digest: root.frozen_project_digest,
        study: root.study.clone(),
        materials: root
            .materials
            .iter()
            .map(|entry| StructuralStudyMaterialExpectation {
                reference: entry.reference.clone(),
                material: entry.material.clone(),
            })
            .collect(),
        planning: root.planning,
        local_execution: root.local_execution,
        analysis_accelerator_requests: root.analysis_accelerator_requests.clone(),
        input_materializations: input_expectations,
        initial_fields,
        geometry_request,
        expected_meshing_request,
        accelerator_placement: Default::default(),
    };
    expectation.validate()?;
    Ok(PreparedStructuralStudyRoot {
        request,
        objects,
        input_value_objects: input_value_objects.into_values().collect(),
        required_programs,
        authored_study_digest: root.authored_study_digest,
        expectation,
    })
}
