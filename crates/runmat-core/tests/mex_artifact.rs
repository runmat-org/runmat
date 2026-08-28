#![cfg(not(target_family = "wasm"))]

use std::collections::BTreeSet;
use std::process::Command;

use runmat_core::RunMatSession;
use runmat_package::{FrozenProjectHandoff, HostCapability};
use runmat_value::Value;

#[test]
fn frozen_project_installs_exact_mex_artifact_outside_the_search_path() {
    if Command::new(if cfg!(windows) { "gcc" } else { "clang" })
        .arg("--version")
        .output()
        .is_err()
    {
        eprintln!("project MEX integration requires a C compiler");
        return;
    }
    let fixture = tempfile::tempdir().unwrap();
    let project = fixture.path().join("project");
    let artifacts = project.join("artifacts");
    std::fs::create_dir_all(project.join("src")).unwrap();
    std::fs::create_dir_all(&artifacts).unwrap();
    let source = fixture.path().join("project_fixture.c");
    std::fs::write(
        &source,
        r#"
#include "mex.h"
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    (void)nrhs; (void)prhs;
    if (nlhs > 0) plhs[0] = mxCreateDoubleScalar(42.0);
}
"#,
    )
    .unwrap();
    let artifact = runmat_mex::MexBuild::new(&source, &artifacts)
        .compile()
        .unwrap();
    let module = artifact.module.strip_prefix(&project).unwrap();
    let manifest = artifact.manifest.strip_prefix(&project).unwrap();
    std::fs::write(
        project.join("runmat.toml"),
        format!(
            "[package]\nname = \"project-mex\"\n[sources]\nroots = [\"src\"]\n[mex-artifacts.project_fixture]\nmanifest = \"{}\"\nmodule = \"{}\"\n",
            manifest.display(),
            module.display()
        ),
    )
    .unwrap();
    let frozen = runmat_package::build_frozen_project(
        &project.join("runmat.toml"),
        BTreeSet::from([HostCapability::Mex]),
    )
    .unwrap();

    let mut session = RunMatSession::with_options(false, false).unwrap();
    session
        .install_project_handoff(FrozenProjectHandoff::new(frozen))
        .unwrap();
    let result = runmat_core::execute_text_request_for_testing(
        &mut session,
        "value = project_fixture(); value",
    )
    .unwrap();

    assert!(result.error.is_none(), "{:?}", result.error);
    assert_eq!(result.value, Some(Value::Num(42.0)));
}
