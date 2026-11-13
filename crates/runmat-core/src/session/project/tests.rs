use super::*;
use runmat_package::{FrozenProjectHandoff, HostCapability};
use runmat_runtime::foreign::MexNameSelection;
use std::collections::BTreeSet;
use std::process::Command;

mod started_project_fixture {
    use crate as runmat_core;
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/support/started_python_project.rs"
    ));
}
use started_project_fixture::{artifact_project, project_runtime};

fn project_foreign_integer(
    session: &RunMatSession,
    runtime: &tokio::runtime::Runtime,
    adapter: runmat_types::ForeignAdapterId,
    operation: &str,
    arguments: Vec<runmat_value::Value>,
) -> i32 {
    let context = runmat_runtime::context::RuntimeContext::new(std::rc::Rc::new(
        runmat_runtime::execution::RuntimeExecutionService::new(),
    ))
    .with_service_ports(
        runmat_runtime::context::RuntimeServicePorts::default()
            .with_foreign(session.foreign_runtime.clone()),
    );
    let service = context
        .service_ports()
        .require_foreign(operation)
        .expect("registered project foreign service")
        .clone();
    let sequence = runtime
        .block_on(context.scope(service.invoke(
            context.clone(),
            runmat_runtime::context::ForeignCall {
                adapter: adapter.to_string(),
                symbol: operation.into(),
                arguments: runmat_runtime::call::RootedFunctionArguments::ordinary(
                    runmat_runtime::execution::RootedValues::new(arguments).unwrap(),
                ),
                requested_outputs: 1,
            },
        )))
        .into_invoked_result()
        .expect("invoke started project foreign host");
    sequence.with_owned(|sequence| {
        let values = runmat_runtime::sequence::ResolveValueSequence::resolve(
            sequence,
            runmat_types::SequenceUse::RequireSingle,
            runmat_runtime::sequence::SequenceResolutionContext::default(),
        )
        .expect("one project foreign result");
        let value = values
            .try_into_first()
            .expect("unique project foreign output")
            .expect("one project foreign value");
        let runmat_value::Value::Int(value) = value else {
            panic!("project foreign answer must be an exact integer: {value:?}");
        };
        value.try_to_i32().expect("small exact fixture integer")
    })
}

fn python_project_answer(session: &RunMatSession, runtime: &tokio::runtime::Runtime) -> i32 {
    project_foreign_integer(
        session,
        runtime,
        runmat_types::ForeignAdapterId::new(runmat_python::PYTHON_ADAPTER_ID)
            .expect("canonical Python adapter identity"),
        "invoke_qualified",
        vec![runmat_value::Value::String("fixture.answer".into())],
    )
}

#[test]
fn started_python_project_replacement_preserves_old_in_process_lane() {
    started_project_fixture::run_started_python_project(
        runmat_python::PythonExecutionMode::InProcess,
        |session, runtime| python_project_answer(session, runtime),
        |session| {
            assert!(session.python_adapter.is_running());
            session.python_adapter.configuration().clone()
        },
        |session, original_config| {
            assert_eq!(*session.python_adapter.configuration(), *original_config);
        },
    );
}

fn jar_with_answer(
    home: &std::path::Path,
    root: &std::path::Path,
    answer: u8,
) -> std::path::PathBuf {
    let source = root.join("Value.java");
    let classes = root.join("classes");
    std::fs::create_dir_all(&classes).unwrap();
    std::fs::write(
        &source,
        format!("package fixture.project; public final class Value {{ public static int answer() {{ return {answer}; }} }}"),
    )
    .unwrap();
    let compiler = home
        .join("bin")
        .join(if cfg!(windows) { "javac.exe" } else { "javac" });
    assert!(Command::new(compiler)
        .arg("-d")
        .arg(&classes)
        .arg(&source)
        .status()
        .expect("run explicit JDK compiler")
        .success());
    let jar = root.join("fixture.jar");
    let archiver = home
        .join("bin")
        .join(if cfg!(windows) { "jar.exe" } else { "jar" });
    assert!(Command::new(archiver)
        .arg("cf")
        .arg(&jar)
        .arg("-C")
        .arg(&classes)
        .arg(".")
        .status()
        .expect("run explicit JDK archiver")
        .success());
    jar
}

fn java_project_answer(session: &RunMatSession, runtime: &tokio::runtime::Runtime) -> i32 {
    project_foreign_integer(
        session,
        runtime,
        runmat_types::ForeignAdapterId::new(runmat_java::JAVA_ADAPTER_ID)
            .expect("canonical Java adapter identity"),
        "call_static",
        vec![
            runmat_value::Value::String("fixture.project.Value".into()),
            runmat_value::Value::String("answer".into()),
        ],
    )
}

#[test]
fn started_java_project_replacement_preserves_old_classloader_then_restarts() {
    let Some(home) = std::env::var_os("RUNMAT_TEST_JAVA_HOME").map(std::path::PathBuf::from) else {
        eprintln!(
            "RUNMAT_TEST_JAVA_HOME is required to qualify started project classloader replacement"
        );
        return;
    };
    let fixture = tempfile::tempdir().expect("Java project fixture");
    let old_root = fixture.path().join("old");
    let new_root = fixture.path().join("new");
    std::fs::create_dir_all(&old_root).unwrap();
    std::fs::create_dir_all(&new_root).unwrap();
    let old_jar = jar_with_answer(&home, &old_root, 17);
    let new_jar = jar_with_answer(&home, &new_root, 29);
    let old = artifact_project(&old_root, HostCapability::Jvm, &old_jar);
    let next = artifact_project(&new_root, HostCapability::Jvm, &new_jar);
    let mut session = RunMatSession::with_options(false, false).expect("Java project session");
    let config = runmat_config::runtime::JavaConfig {
        home: Some(home),
        ..Default::default()
    };
    session
        .set_java_config(&config)
        .expect("explicit JDK policy");
    let runtime = project_runtime();
    runtime
        .block_on(session.install_project_handoff(old))
        .expect("install old Java project");
    assert_eq!(java_project_answer(&session, &runtime), 17);
    assert!(session.java_adapter.is_running());
    let original_revision = session.project_revision();
    let next_bytes = std::fs::read(&new_jar).unwrap();
    std::fs::write(&new_jar, b"invalid JAR after freezing").unwrap();
    assert!(runtime
        .block_on(session.install_project_handoff(next.clone()))
        .is_err());
    assert_eq!(session.project_revision(), original_revision);
    assert_eq!(java_project_answer(&session, &runtime), 17);
    std::fs::write(&new_jar, next_bytes).unwrap();
    runtime
        .block_on(session.install_project_handoff(next))
        .expect("replace started Java classloader");
    assert_eq!(java_project_answer(&session, &runtime), 29);
    runtime
        .block_on(session.clear_project_handoff())
        .expect("clear started Java project");
    assert!(session.project_revision().is_none());
}

#[test]
fn java_callback_reenters_core_source_execution_and_java_lane() {
    let Some(home) = std::env::var_os("RUNMAT_TEST_JAVA_HOME").map(std::path::PathBuf::from) else {
        eprintln!("RUNMAT_TEST_JAVA_HOME is required to qualify Core Java callback reentry");
        return;
    };
    let runtime = project_runtime();
    let source_text = concat!(
        "advance = @(x) javaMethod('sum', 'java.lang.Integer', x, int32(2));\n",
        "stream = javaMethod('iterate', 'java.util.stream.IntStream', int32(1), advance);\n",
        "limited = javaMethod('limit', stream, int64(3));\n",
        "values = javaMethod('toArray', limited);\n",
        "values"
    );
    for enable_jit in [false, true]
        .into_iter()
        .filter(|&enabled| !enabled || cfg!(feature = "jit"))
    {
        for frozen_project in [false, true] {
            let mut session = RunMatSession::with_options(enable_jit, false)
                .expect("independent Core Java callback session");
            session
                .set_java_config(&runmat_config::runtime::JavaConfig {
                    home: Some(home.clone()),
                    ..Default::default()
                })
                .expect("explicit JDK policy");
            let fixture =
                frozen_project.then(|| tempfile::tempdir().expect("jarless Java project fixture"));
            let source = if let Some(fixture) = &fixture {
                let source_dir = fixture.path().join("src");
                std::fs::create_dir_all(&source_dir).expect("project source directory");
                let source_path = source_dir.join("java-callback-reentry.m");
                std::fs::write(
                    &source_path,
                    concat!(
                        "aInner = @(z) javaMethod('sum', 'java.lang.Integer', z, int32(1));\n",
                        "advance = @(x) crossHostB(x, aInner);\n",
                        "stream = javaMethod('iterate', 'java.util.stream.IntStream', int32(1), advance);\n",
                        "limited = javaMethod('limit', stream, int64(3));\n",
                        "values = javaMethod('toArray', limited);\n",
                        "counter = javaObject('java.util.concurrent.atomic.AtomicInteger', int32(0));\n",
                        "support = javaObject('java.beans.PropertyChangeSupport', 'source');\n",
                        "listener = @(event) javaMethod('incrementAndGet', counter);\n",
                        "javaMethod('addPropertyChangeListener', support, listener);\n",
                        "javaMethod('firePropertyChange', support, 'value', int32(1), int32(2));\n",
                        "assert(javaMethod('get', counter) == int32(1));\n",
                        "javaMethod('removePropertyChangeListener', support, listener);\n",
                        "javaMethod('firePropertyChange', support, 'value', int32(2), int32(3));\n",
                        "assert(javaMethod('get', counter) == int32(1));\n",
                        "values"
                    ),
                )
                .expect("project callback source");
                std::fs::write(
                    source_dir.join("crossHostB.m"),
                    concat!(
                        "function y = crossHostB(x, aInner)\n",
                        " innerAdvance = @(z) feval(aInner, z);\n",
                        " inner = javaMethod('iterate', 'java.util.stream.IntStream', int32(1), innerAdvance);\n",
                        " limited = javaMethod('limit', inner, int64(2));\n",
                        " innerValues = javaMethod('toArray', limited);\n",
                        " y = javaMethod('sum', 'java.lang.Integer', x, int32(innerValues(2)));\n",
                        "end\n"
                    ),
                )
                .expect("project nested callback source");
                let manifest_path = fixture.path().join("runmat.toml");
                std::fs::write(
                    &manifest_path,
                    "[package]\nname = \"jarless-java-callback\"\n[sources]\nroots = [\"src\"]\n",
                )
                .expect("jarless project manifest");
                let project = runmat_package::build_frozen_project(
                    &manifest_path,
                    BTreeSet::from([HostCapability::Jvm]),
                )
                .expect("frozen Java source without JAR declarations");
                assert!(project.java_artifacts.is_empty());
                runtime
                    .block_on(session.install_project_handoff(FrozenProjectHandoff::new(project)))
                    .expect("install authentic jarless project handoff");
                let source_path = std::fs::canonicalize(&source_path)
                    .expect("exact canonical frozen project source path");
                crate::abi::SourceInput::Path(source_path.to_string_lossy().into_owned())
            } else {
                crate::abi::SourceInput::Text {
                    name: "java-callback-reentry.m".into(),
                    text: source_text.into(),
                }
            };
            let request = crate::abi::ExecutionRequest::for_source(
                source,
                session.compat_mode(),
                crate::abi::HostExecutionPolicy::default(),
                session.workspace_handle(),
            );
            let response = runtime.block_on(session.execute_request(request));
            let outcome = response
                .result
                .as_ref()
                .expect("Core Java callback outcome");
            assert!(
                outcome.diagnostics.iter().all(|diagnostic| {
                    diagnostic.severity != crate::abi::DiagnosticSeverity::Error
                }),
                "Java callback diagnostics (JIT requested: {enable_jit}, frozen project: {frozen_project}): {:?}",
                outcome.diagnostics
            );
            assert!(
                outcome.suspension.is_none(),
                "Java callback did not finish (JIT requested: {enable_jit}, frozen project: {frozen_project}): {:?}",
                outcome.suspension
            );
            let Some(runmat_value::Value::Tensor(values)) = outcome.flow.durable_workspace_value()
            else {
                panic!(
                    "Java callback must return a native integer tensor (JIT requested: {enable_jit}, frozen project: {frozen_project}): {:?}",
                    outcome.flow
                );
            };
            assert_eq!(values.shape(), &[1, 3]);
            assert_eq!(
                values.integer_storage(),
                Some(&runmat_value::IntegerStorage::I32(vec![1, 3, 5]))
            );
        }
    }
}

#[test]
fn pure_handoff_preserves_unrelated_java_and_python_owners() {
    let fixture = tempfile::tempdir().expect("pure project fixture");
    std::fs::create_dir_all(fixture.path().join("src")).expect("pure source directory");
    let source_text = "function value = primary(); value = 1; end\n";
    std::fs::write(fixture.path().join("src/primary.m"), source_text)
        .expect("primary project source");
    std::fs::write(
        fixture.path().join("runmat.toml"),
        "[package]\nname = \"pure-handoff\"\n[sources]\nroots = [\"src\"]\n",
    )
    .expect("pure project manifest");
    let handoff = FrozenProjectHandoff::new(
        runmat_package::build_frozen_project(&fixture.path().join("runmat.toml"), BTreeSet::new())
            .expect("pure frozen project"),
    );
    let source_id = handoff
        .project
        .all_sources()
        .next()
        .expect("one captured project source")
        .0
        .id
        .clone();
    let source = crate::ExecutableSource::from_project_source(&source_id, source_text)
        .expect("exact captured source ownership");
    assert!(crate::ExecutableSource::from_project_source(&source_id, "changed = 2;").is_err());
    let mut session = RunMatSession::with_options(false, false).expect("pure session");
    // These owners reject a project replacement if one is staged for them.
    session
        .java_adapter
        .shutdown_selected_retirement()
        .expect("mark unrelated Java owner retired");
    let python_adapter = std::rc::Rc::clone(&session.python_adapter);
    let python_configuration = python_adapter.configuration();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("pure project runtime");
    runtime
        .block_on(session.install_project_handoff(handoff))
        .expect("pure handoff leaves unrelated adapters alone");
    runtime
        .block_on(session.compile_executable_unit(source, None))
        .expect("primary source compiles under its captured project owner");
    assert!(session
        .java_adapter
        .prepare_project_replacement(&[])
        .is_err());
    assert_eq!(&*python_adapter.configuration(), &*python_configuration);
}

#[test]
fn failed_project_retirement_preserves_revision_and_installed_mex_artifact() {
    if Command::new(if cfg!(windows) { "gcc" } else { "clang" })
        .arg("--version")
        .output()
        .is_err()
    {
        eprintln!("project retirement fixture requires a C compiler");
        return;
    }
    let fixture = tempfile::tempdir().expect("project fixture");
    let project = fixture.path().join("project");
    let artifacts = project.join("artifacts");
    std::fs::create_dir_all(project.join("src")).expect("source directory");
    std::fs::create_dir_all(&artifacts).expect("artifact directory");
    let source = fixture.path().join("handoff_probe.c");
    std::fs::write(
        &source,
        r#"
#include "mex.h"
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    (void)nrhs; (void)prhs;
    if (nlhs > 0) plhs[0] = mxCreateDoubleScalar(7.0);
}
"#,
    )
    .expect("MEX source");
    let artifact = runmat_mex::MexBuild::new(&source, &artifacts)
        .compile()
        .expect("build MEX fixture");
    let module = artifact
        .module
        .strip_prefix(&project)
        .expect("project module");
    let manifest = artifact
        .manifest
        .strip_prefix(&project)
        .expect("project manifest");
    let manifest_literal = toml::Value::String(manifest.to_string_lossy().into_owned()).to_string();
    let module_literal = toml::Value::String(module.to_string_lossy().into_owned()).to_string();
    std::fs::write(
        project.join("runmat.toml"),
        format!(
            "[package]\nname = \"project-retirement\"\n[sources]\nroots = [\"src\"]\n[mex-artifacts.handoff_probe]\nmanifest = {manifest_literal}\nmodule = {module_literal}\n"
        ),
    )
    .expect("project manifest source");
    let frozen = runmat_package::build_frozen_project(
        &project.join("runmat.toml"),
        BTreeSet::from([HostCapability::Mex]),
    )
    .expect("frozen MEX project");
    let handoff = FrozenProjectHandoff::new(frozen);
    let revision = handoff.revision();
    let mut session = RunMatSession::with_options(false, false).expect("session");
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("project lifecycle runtime");
    runtime
        .block_on(session.install_project_handoff(handoff.clone()))
        .expect("install project and its MEX artifact");
    let installed_path = session
        .project_handoff()
        .expect("published project")
        .project
        .mex_artifacts[0]
        .module_path
        .clone();
    assert_ne!(installed_path, artifact.module);
    assert!(installed_path.starts_with(
        session
            .private_project_root
            .as_ref()
            .expect("private project byte owner")
            .path()
    ));
    assert_eq!(
        std::fs::read(&installed_path).expect("private MEX bytes"),
        std::fs::read(&artifact.module).expect("original MEX bytes")
    );
    let selected = runtime
        .block_on(session.mex_runtime.observe_name("handoff_probe"))
        .expect("observe installed MEX");
    assert!(matches!(selected, Some(MexNameSelection::Installed { .. })));

    let invalid_wheel = artifacts.join("later_bad.whl");
    std::fs::write(&invalid_wheel, b"not a wheel archive").expect("bad later artifact");
    let mut bad_later = handoff.clone();
    bad_later
        .project
        .python_artifacts
        .push(runmat_package::FrozenPythonArtifact {
            package_instance: bad_later.project.mex_artifacts[0].package_instance.clone(),
            name: "later_bad".into(),
            module: "later_bad".into(),
            digest: runmat_package::ContentDigest::sha256(b"not a wheel archive"),
            path: invalid_wheel,
        });
    let late_error = runtime
        .block_on(session.install_project_handoff(bad_later))
        .expect_err("bad later Python artifact must fail before retiring installed MEX");
    assert!(late_error
        .to_string()
        .contains("could not install frozen project artifact"));
    assert_eq!(session.project_revision(), Some(revision.clone()));
    assert_eq!(
        &session.project_handoff().unwrap().project.mex_artifacts[0].module_path,
        &installed_path
    );
    assert!(matches!(
        runtime
            .block_on(session.mex_runtime.observe_name("handoff_probe"))
            .expect("observe old MEX after rejected replacement"),
        Some(MexNameSelection::Installed { .. })
    ));

    session.shutdown_local_tasks();
    let install_error = runtime
        .block_on(session.install_project_handoff(handoff))
        .expect_err("closed host cannot retire old MEX before replacement");
    assert!(install_error
        .to_string()
        .contains("could not clear loaded MEX modules"));
    let clear_error = runtime
        .block_on(session.clear_project_handoff())
        .expect_err("closed host cannot retire MEX before clearing project");
    assert!(clear_error
        .to_string()
        .contains("could not clear loaded MEX modules"));
    assert_eq!(session.project_revision(), Some(revision));
    assert!(session.project_handoff().is_some());
    let selected = runtime
        .block_on(session.mex_runtime.observe_name("handoff_probe"))
        .expect("observe preserved installed MEX");
    assert!(matches!(selected, Some(MexNameSelection::Installed { .. })));
}
