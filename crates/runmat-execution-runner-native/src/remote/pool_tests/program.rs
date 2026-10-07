use super::*;

pub(super) async fn build(
    source_text: &str,
    environment: Vec<(String, String)>,
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
        .unwrap();
    let unit = session
        .compile_executable_unit(
            runmat_core::ExecutableSource::new("root", "src/answer.m", source_text),
            Some(revision.clone()),
        )
        .await
        .unwrap();
    let envelope = unit.portable_envelope_for(Some("answer")).unwrap();
    let function = usize::try_from(envelope.manifest.identity.entrypoint_function.0).unwrap();
    let recipe = ProgramBuildRecipe {
        schema_version: runmat_execution_artifact::PROGRAM_BUILD_RECIPE_SCHEMA_VERSION,
        program_revision: revision.clone(),
        entrypoint: function.to_string(),
        outputs: OutputContract {
            requested_outputs: 1,
        },
        execution_mode: "interpreter".into(),
        target: runmat_execution_artifact::ProgramTarget::portable("remote-pool-test"),
        features: BTreeSet::new(),
        compile_options: BTreeSet::new(),
        source_objects: Vec::new(),
        expected_artifact_id: None,
    };
    let executable_bytes = envelope.canonical_bytes().unwrap();
    let artifact =
        ProgramArtifact::materialize(&recipe, ExecutableForm::ExecutableUnitV4, executable_bytes)
            .unwrap();
    let bundle = ExecutionBundleBuilder::native(&project, recipe.program_revision.clone())
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
    assert!(bundle.objects.is_empty());
    assert!(bundle.manifest.sources.is_empty());
    let recipe = bundle.manifest.recipes.first().cloned().unwrap();
    let artifact = bundle.manifest.artifacts.first().cloned().unwrap();
    let artifact_id = ArtifactId::derive(&[artifact.id.0.bytes()]);
    let mut bundle_bytes = Vec::new();
    write_bundle(&bundle, &mut bundle_bytes, ArchiveLimits::default()).unwrap();
    let program = ProgramExecutionRequest {
        environment: runmat_execution::EnvironmentSnapshot::new(
            bundle.manifest.portable_environment.clone(),
        )
        .unwrap(),
        schema_version: PROGRAM_EXECUTION_REQUEST_SCHEMA_V2,
        recipe,
        artifact,
        function,
        arguments: Vec::new(),
        requested_outputs: 1,
    };
    (program, artifact_id, bundle_bytes)
}
