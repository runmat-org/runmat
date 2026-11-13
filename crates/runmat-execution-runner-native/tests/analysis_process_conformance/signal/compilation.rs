//! One frozen-project fixture compiler for named and capturing signal placement checks.
use super::*;

const PROJECT_DOCUMENT: &str = "[package]\nname = \"signal-placement-conformance\"\n[sources]\nroots = [\"src\"]\n[runtime.language]\ncompat = \"matlab\"\n";

pub(super) fn project_policy() -> runmat_config::runtime::ClusterRuntimePolicy {
    runmat_config::runtime::ClusterRuntimePolicy::from_project_document(
        PROJECT_DOCUMENT,
        runmat_config::document::RunmatConfigFormat::Toml,
    )
    .unwrap()
}

pub(super) struct CompiledSignalFixture {
    pub session: runmat_core::RunMatSession,
    pub project_root: tempfile::TempDir,
    pub project: runmat_package::FrozenProject,
    pub revision: ProgramRevision,
    pub unit: runmat_core::ExecutableUnit,
    pub owner_identity: String,
}

pub(super) async fn compile(source: &str) -> CompiledSignalFixture {
    let project_root = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(project_root.path().join("src")).unwrap();
    std::fs::write(project_root.path().join("runmat.toml"), PROJECT_DOCUMENT).unwrap();
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
        session,
        project_root,
        project,
        revision,
        unit,
        owner_identity,
    }
}

pub(super) async fn snapshot_factory(
    session: &runmat_core::RunMatSession,
    unit: &runmat_core::ExecutableUnit,
    revision: &ProgramRevision,
) -> runmat_execution::value::PayloadGraph {
    let (local, _) = unit
        .mir()
        .functions
        .iter()
        .find(|(_, function)| function.source_role == runmat_types::FunctionSourceRole::Primary)
        .expect("original factory primary function");
    let factory = unit
        .analysis()
        .function_for_local(*local)
        .expect("exact factory semantic owner")
        .function;
    let envelope = unit.portable_envelope_for_function(factory).unwrap();
    let admitted = runmat_vm::admit_executable_unit(&envelope.canonical_bytes().unwrap()).unwrap();
    let runtime = session.execution_runtime_context(revision.clone());
    let bound = admitted.bind_runtime_sources(runtime).unwrap();
    let values = runmat_vm::invoke_semantic_function_value_in_context(
        bound.entrypoint_function().unwrap().0,
        &[],
        1,
        bound.registry(),
        bound.runtime().clone(),
    )
    .await
    .unwrap()
    .try_into_single()
    .unwrap();
    let runmat_value::Value::Closure(closure) = values.value() else {
        panic!("original makePulse must construct one nested closure");
    };
    assert!(
        closure.lexical_environment.is_some(),
        "actual compiler-issued shared activation"
    );
    assert!(
        runmat_runtime::execution::value_codec::encode_inline_value(values.value()).is_err(),
        "shared activation must not be flattened through the legacy value codec"
    );
    let graph = bound
        .graph_codec_admission()
        .unwrap()
        .snapshot(
            std::slice::from_ref(values.value()),
            runmat_execution::value::ValueLimits::default(),
        )
        .unwrap();
    assert!(graph
        .nodes
        .iter()
        .any(|node| matches!(node, runmat_execution::value::GraphNode::Activation(_))));
    // The rooted factory output and exact selected source owner remain alive
    // through the complete snapshot; no raw closure/cell is returned.
    graph
}
