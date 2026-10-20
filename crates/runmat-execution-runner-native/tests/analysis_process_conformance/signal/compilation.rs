//! One frozen-project fixture compiler for named and capturing signal placement checks.
use super::*;

pub(super) struct CompiledSignalFixture {
    pub project_root: tempfile::TempDir,
    pub project: runmat_package::FrozenProject,
    pub revision: ProgramRevision,
    pub unit: runmat_core::ExecutableUnit,
    pub owner_identity: String,
}

pub(super) async fn compile(source: &str) -> CompiledSignalFixture {
    let project_root = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(project_root.path().join("src")).unwrap();
    std::fs::write(
        project_root.path().join("runmat.toml"),
        "[package]\nname = \"signal-placement-conformance\"\n[sources]\nroots = [\"src\"]\n",
    )
    .unwrap();
    let source_path = project_root.path().join("src/pulse.m");
    std::fs::write(&source_path, source).unwrap();
    let project = runmat_package::build_frozen_project(
        &project_root.path().join("runmat.toml"),
        BTreeSet::new(),
    )
    .unwrap();
    let project_identity = project.revision();
    let revision = ProgramRevision::new(
        Digest::from_bytes(*project_identity.graph_digest.bytes()),
        Digest::from_bytes(*project_identity.source_revision.bytes()),
        runmat_core::program_environment(runmat_core::CompatMode::Matlab),
    )
    .unwrap();
    let symbols = runmat_package::source_symbols_from_frozen(&project, &source_path);
    let definition = symbols
        .definitions
        .iter()
        .find(|definition| definition.name == "pulse")
        .unwrap();
    let source_id = definition.source_id.as_ref().unwrap();
    let owner_identity = source_id.package_instance.to_string();
    let mut session = runmat_core::RunMatSession::with_options(false, false).unwrap();
    session
        .install_project_handoff(runmat_package::FrozenProjectHandoff::new(project.clone()))
        .await
        .unwrap();
    let unit = session
        .compile_executable_unit(
            runmat_core::ExecutableSource::frozen(
                owner_identity.clone(),
                source_id.relative_path.as_str(),
                source_path.to_string_lossy(),
                source,
            ),
            Some(revision.clone()),
        )
        .await
        .unwrap();
    CompiledSignalFixture {
        project_root,
        project,
        revision,
        unit,
        owner_identity,
    }
}
