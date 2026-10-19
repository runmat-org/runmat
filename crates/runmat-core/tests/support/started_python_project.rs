use runmat_core::RunMatSession;
use runmat_package::{FrozenProjectHandoff, HostCapability};
use std::collections::BTreeSet;
use std::path::Path;
use std::process::Command;

pub(super) fn artifact_project(
    root: &Path,
    family: HostCapability,
    artifact_path: &Path,
) -> FrozenProjectHandoff {
    let (section, module) = match family {
        HostCapability::Jvm => ("java-artifacts.fixture", ""),
        HostCapability::Python => ("python-artifacts.fixture", "module = \"fixture\"\n"),
        _ => panic!("started project fixture supports only Java and Python"),
    };
    std::fs::create_dir_all(root.join("src")).expect("project source directory");
    std::fs::write(
        root.join("src/primary.m"),
        "function y = primary(); y = 1; end\n",
    )
    .expect("project source");
    std::fs::write(
        root.join("runmat.toml"),
        format!(
            "[package]\nname = \"started-foreign\"\n[sources]\nroots = [\"src\"]\n[{section}]\npath = \"{}\"\n{module}",
            artifact_path.file_name().unwrap().to_string_lossy(),
        ),
    )
    .expect("project manifest");
    FrozenProjectHandoff::new(
        runmat_package::build_frozen_project(&root.join("runmat.toml"), BTreeSet::from([family]))
            .expect("frozen project artifact"),
    )
}

pub(super) fn project_runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("started foreign project runtime")
}

fn wheel_with_answer(python: &Path, path: &Path, answer: u8) {
    let status = Command::new(python)
        .arg("-c")
        .arg(
            "import sys, zipfile\nwith zipfile.ZipFile(sys.argv[1], 'w') as wheel:\n wheel.writestr('fixture/__init__.py', 'def answer():\\n return ' + sys.argv[2] + '\\n')\n wheel.writestr('fixture-1.dist-info/METADATA', 'Name: fixture\\nVersion: 1\\n')",
        )
        .arg(path)
        .arg(answer.to_string())
        .status()
        .expect("run discovered Python wheel fixture builder");
    assert!(
        status.success(),
        "discovered Python must build its fixture wheel"
    );
}

pub(super) fn run_started_python_project<Answer, Started, Failed, Snapshot>(
    mode: runmat_python::PythonExecutionMode,
    mut answer: Answer,
    on_started: Started,
    on_failed_replacement: Failed,
) where
    Answer: FnMut(&mut RunMatSession, &tokio::runtime::Runtime) -> i32,
    Started: FnOnce(&RunMatSession) -> Snapshot,
    Failed: FnOnce(&RunMatSession, &Snapshot),
{
    let installation =
        match runmat_python::discover_python(&runmat_python::PythonDiscoveryRequest::default()) {
            Ok(installation) => installation,
            Err(_) => {
                eprintln!("started Python project fixture requires a discoverable Python");
                return;
            }
        };
    let fixture = tempfile::tempdir().expect("Python project fixture");
    let old_root = fixture.path().join("old");
    let new_root = fixture.path().join("new");
    std::fs::create_dir_all(&old_root).unwrap();
    std::fs::create_dir_all(&new_root).unwrap();
    let old_wheel = old_root.join("fixture-1-py3-none-any.whl");
    let new_wheel = new_root.join("fixture-1-py3-none-any.whl");
    wheel_with_answer(&installation.executable, &old_wheel, 17);
    wheel_with_answer(&installation.executable, &new_wheel, 29);
    let old = artifact_project(&old_root, HostCapability::Python, &old_wheel);
    let next = artifact_project(&new_root, HostCapability::Python, &new_wheel);
    let mut session = RunMatSession::with_options(false, false).expect("Python project session");
    let config = runmat_runtime::foreign::PythonRuntimeConfiguration {
        executable: Some(installation.executable.clone()),
        execution_mode: mode,
        ..Default::default()
    };
    session
        .configure_python_runtime(config)
        .expect("configure Python mode");
    let runtime = project_runtime();
    runtime
        .block_on(session.install_project_handoff(old.clone()))
        .expect("install old Python project");
    assert_eq!(answer(&mut session, &runtime), 17);
    let started = on_started(&session);
    let original_revision = session.project_revision();
    let next_bytes = std::fs::read(&new_wheel).unwrap();
    std::fs::write(&new_wheel, b"invalid wheel after freezing").unwrap();
    assert!(runtime
        .block_on(session.install_project_handoff(next.clone()))
        .is_err());
    assert_eq!(session.project_revision(), original_revision);
    on_failed_replacement(&session, &started);
    assert_eq!(answer(&mut session, &runtime), 17);
    std::fs::write(&new_wheel, next_bytes).unwrap();
    runtime
        .block_on(session.install_project_handoff(next))
        .expect("replace started Python project");
    assert_eq!(answer(&mut session, &runtime), 29);
    runtime
        .block_on(session.clear_project_handoff())
        .expect("clear started Python project");
    assert!(session.project_revision().is_none());
}
