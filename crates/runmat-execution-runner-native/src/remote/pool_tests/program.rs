use super::*;

pub(super) async fn build(
    source_text: &str,
    environment: Vec<(String, String)>,
    accelerators: Vec<AcceleratorRequest>,
) -> (ProgramExecutionRequest, ArtifactId, Vec<u8>) {
    let project_root = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(project_root.path().join("src")).unwrap();
    std::fs::write(
        project_root.path().join("runmat.toml"),
        "[package]\nname = \"remote-pool\"\n[sources]\nroots = [\"src\"]\n",
    )
    .unwrap();
    std::fs::write(project_root.path().join("src/answer.m"), source_text).unwrap();
    let project = runmat_package::build_frozen_project(
        &project_root.path().join("runmat.toml"),
        BTreeSet::new(),
    )
    .unwrap();
    let project_revision = project.revision();
    let revision = ProgramRevision::new(
        Digest::from_bytes(*project_revision.graph_digest.bytes()),
        Digest::from_bytes(*project_revision.source_revision.bytes()),
        runmat_core::program_environment(runmat_core::CompatMode::Matlab),
    )
    .unwrap();
    let mut session = runmat_core::RunMatSession::with_options(false, false).unwrap();
    session
        .install_project_handoff(runmat_package::FrozenProjectHandoff::new(project.clone()))
        .await
        .unwrap();
    let unit = session
        .compile_executable_unit(
            runmat_core::ExecutableSource::new(
                project.graph.root.to_string(),
                "src/answer.m",
                source_text,
            ),
            Some(revision.clone()),
        )
        .await
        .unwrap();
    let envelope = unit.portable_envelope_for(Some("answer")).unwrap();
    let function = envelope
        .manifest
        .identity
        .entrypoint_function
        .expect("answer fixture has a callable entrypoint");
    let recipe = ProgramBuildRecipe {
        schema_version: runmat_execution_artifact::PROGRAM_BUILD_RECIPE_SCHEMA_VERSION,
        program_revision: revision.clone(),
        entrypoint: envelope.manifest.identity.entrypoint.clone(),
        outputs: OutputContract {
            requested_outputs: 1,
            owned_work: None,
        },
        execution_mode: "interpreter".into(),
        target: runmat_execution_artifact::ProgramTarget::portable("remote-pool-test"),
        interop: runmat_types::InteropManifest::empty(),
        accelerators,
        features: BTreeSet::new(),
        compile_options: BTreeSet::new(),
        source_objects: Vec::new(),
        foreign_policy: None,
        expected_artifact_id: None,
    };
    let executable_bytes = envelope.canonical_bytes().unwrap();
    let artifact =
        ProgramArtifact::materialize(&recipe, ExecutableForm::ExecutableUnitV4, executable_bytes)
            .unwrap();
    let bundle = ExecutionBundleBuilder::native(&project, recipe.program_revision.clone())
        .unwrap()
        .with_project_runtime_policy_object(crate::remote::runtime_settings::fixture_policy(
            &recipe.program_revision,
        ))
        .unwrap()
        .with_compiled_package_closure()
        .with_materialized_program(
            recipe.clone(),
            ExecutableForm::ExecutableUnitV4,
            artifact.executable_bytes.clone(),
        )
        .with_portable_environment(environment)
        .unwrap()
        .build()
        .unwrap();
    assert!(!bundle.requires_source_project());
    assert_eq!(bundle.objects.len(), 1);
    assert!(bundle.project_runtime_policy_object().unwrap().is_some());
    assert!(bundle.manifest.sources.is_empty());
    let recipe = bundle.manifest.recipes.first().cloned().unwrap();
    let artifact = bundle.manifest.artifacts.first().cloned().unwrap();
    let artifact_id = ArtifactId::derive(&[artifact.id.0.bytes()]);
    let mut bundle_bytes = Vec::new();
    write_bundle(&bundle, &mut bundle_bytes, ArchiveLimits::default()).unwrap();
    let program = ProgramExecutionRequest {
        console: None,
        environment: runmat_execution::EnvironmentSnapshot::new(
            bundle.manifest.portable_environment.clone(),
        )
        .unwrap(),
        schema_version: PROGRAM_EXECUTION_REQUEST_SCHEMA_VERSION,
        recipe,
        artifact,
        callable: ProgramCallable::semantic(function, None),
        context: ProgramInvocationContext::Direct,
        assignment: None,
        job_id: None,
        arguments: Vec::new(),
        requested_outputs: 1,
    };
    (program, artifact_id, bundle_bytes)
}
