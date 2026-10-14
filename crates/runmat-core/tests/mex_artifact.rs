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
    let marker = fixture.path().join("project_at_exit.marker");
    let marker_literal = marker
        .to_string_lossy()
        .replace('\\', "\\\\")
        .replace('"', "\\\"");
    std::fs::write(
        &source,
        format!(
            r#"
#include "mex.h"
#include <stdio.h>
static void project_at_exit(void) {{
    FILE *file = fopen("{marker_literal}", "w");
    if (file != NULL) {{ fputs("called", file); fclose(file); }}
}}
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {{
    (void)nrhs; (void)prhs;
    mexAtExit(project_at_exit);
    if (nlhs > 0) plhs[0] = mxCreateDoubleScalar(42.0);
}}
"#
        ),
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
    let handoff = FrozenProjectHandoff::new(frozen);
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime
        .block_on(session.install_project_handoff(handoff.clone()))
        .unwrap();
    let result = {
        let _entered = runtime.enter();
        runmat_core::execute_text_request_for_testing(
            &mut session,
            "value = project_fixture(); value",
        )
        .unwrap()
    };

    assert!(result.error.is_none(), "{:?}", result.error);
    assert_eq!(result.value, Some(Value::Num(42.0)));
    assert!(
        !marker.exists(),
        "MEX module remains live before project clear"
    );
    drop(result);
    {
        let _entered = runtime.enter();
        let captured = runmat_core::execute_text_request_for_testing(
            &mut session,
            "selected_mex = @() project_fixture();",
        )
        .unwrap();
        assert!(captured.error.is_none(), "{:?}", captured.error);
        let ordinary = runmat_core::execute_text_request_for_testing(
            &mut session,
            "ordinary_mex_request = 1;",
        )
        .unwrap();
        assert!(ordinary.error.is_none(), "{:?}", ordinary.error);
        let invoked = runmat_core::execute_text_request_for_testing(
            &mut session,
            "selected_value = selected_mex(); selected_value",
        )
        .unwrap();
        assert!(invoked.error.is_none(), "{:?}", invoked.error);
        assert_eq!(invoked.value, Some(Value::Num(42.0)));
    }
    let revision = session.project_revision();
    let manifest_bytes = std::fs::read(&artifact.manifest).unwrap();
    std::fs::write(&artifact.manifest, b"invalid MEX manifest").unwrap();
    let rejected = runtime
        .block_on(session.install_project_handoff(handoff.clone()))
        .expect_err("invalid replacement must not retire the live project");
    assert!(rejected
        .to_string()
        .contains("MEX manifest changed after project resolution"));
    assert_eq!(session.project_revision(), revision);
    assert!(
        !marker.exists(),
        "failed preparation must not run mexAtExit"
    );
    std::fs::write(&artifact.manifest, manifest_bytes).unwrap();
    {
        let _entered = runtime.enter();
        let preserved = runmat_core::execute_text_request_for_testing(
            &mut session,
            "preserved_mex_value = selected_mex(); preserved_mex_value",
        )
        .unwrap();
        assert!(preserved.error.is_none(), "{:?}", preserved.error);
        assert_eq!(preserved.value, Some(Value::Num(42.0)));
    }
    runtime
        .block_on(session.install_project_handoff(handoff))
        .expect("reinstall waits for old project MEX retirement");
    assert_eq!(std::fs::read_to_string(&marker).unwrap(), "called");
    std::fs::remove_file(&marker).unwrap();
    let result = {
        let _entered = runtime.enter();
        runmat_core::execute_text_request_for_testing(
            &mut session,
            "value = project_fixture(); value",
        )
        .unwrap()
    };
    assert!(result.error.is_none(), "{:?}", result.error);
    assert_eq!(result.value, Some(Value::Num(42.0)));
    assert!(
        !marker.exists(),
        "reinstalled MEX remains live before clear"
    );
    drop(result);
    runtime
        .block_on(session.clear_project_handoff())
        .expect("retire project MEX");
    assert_eq!(session.project_revision(), None);
    assert_eq!(std::fs::read_to_string(marker).unwrap(), "called");
}
