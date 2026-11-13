#![cfg(not(target_family = "wasm"))]

use std::collections::BTreeSet;
use std::process::Command;

use runmat_core::RunMatSession;
use runmat_package::{FrozenProjectHandoff, HostCapability};
use runmat_value::Value;

#[test]
fn ordinary_source_pins_exact_mex_until_clear_and_revalidates_on_readmission() {
    if Command::new(if cfg!(windows) { "gcc" } else { "clang" })
        .arg("--version")
        .output()
        .is_err()
    {
        eprintln!("ordinary MEX integration requires a C compiler");
        return;
    }
    let fixture = tempfile::tempdir().unwrap();
    let source = fixture.path().join("ordinary_add_one.c");
    std::fs::write(
        &source,
        r#"
#include "mex.h"
static int calls = 0;
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    calls++;
    if (nlhs > 0 && nrhs > 0) plhs[0] = mxCreateDoubleScalar(mxGetScalar(prhs[0]) + calls);
}
"#,
    )
    .unwrap();
    let artifact = runmat_mex::MexBuild::new(&source, fixture.path())
        .compile()
        .unwrap();
    std::fs::write(
        fixture.path().join("ordinary_mex_helper.m"),
        "function y = ordinary_mex_helper(x)\ny = ordinary_add_one(x);\nend\n",
    )
    .unwrap();
    let quoted_path = fixture
        .path()
        .display()
        .to_string()
        .replace('\\', "\\\\")
        .replace('\'', "''");
    let mut session = RunMatSession::with_options(false, false).unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let first = {
        let _entered = runtime.enter();
        runmat_core::execute_text_request_for_testing(
            &mut session,
            &format!("addpath('{quoted_path}'); answer = ordinary_add_one(41); answer"),
        )
        .unwrap()
    };
    assert!(first.error.is_none(), "{:?}", first.error);
    assert_eq!(first.value, Some(Value::Num(42.0)));
    let second = {
        let _entered = runtime.enter();
        runmat_core::execute_text_request_for_testing(
            &mut session,
            "answer = ordinary_add_one(42); answer",
        )
        .unwrap()
    };
    assert!(second.error.is_none(), "{:?}", second.error);
    assert_eq!(second.value, Some(Value::Num(44.0)));
    let helper = {
        let _entered = runtime.enter();
        runmat_core::execute_text_request_for_testing(
            &mut session,
            "answer = ordinary_mex_helper(44); answer",
        )
        .unwrap()
    };
    assert!(helper.error.is_none(), "{:?}", helper.error);
    assert_eq!(helper.value, Some(Value::Num(47.0)));

    // A search-path alias to the same canonical bytes does not become an
    // admitted callable with a different name on a cache hit.
    #[cfg(unix)]
    {
        let alias = artifact.module.with_file_name(format!(
            "ordinary_alias.{}",
            artifact.module.extension().unwrap().to_string_lossy()
        ));
        std::os::unix::fs::symlink(&artifact.module, alias).unwrap();
        let rejected_alias = {
            let _entered = runtime.enter();
            runmat_core::execute_text_request_for_testing(
                &mut session,
                "answer = ordinary_alias(1); answer",
            )
            .unwrap()
        };
        assert_eq!(
            rejected_alias
                .error
                .as_ref()
                .and_then(|error| error.identifier()),
            Some("RunMat:MEX:Artifact")
        );
    }

    // Search-path generation is a selection fact, not module lifetime. An
    // unrelated addpath must not rematerialize the loaded module or reset its
    // static state.
    let unrelated = tempfile::tempdir().unwrap();
    let unrelated_path = unrelated
        .path()
        .display()
        .to_string()
        .replace('\\', "\\\\")
        .replace('\'', "''");
    let unchanged = {
        let _entered = runtime.enter();
        runmat_core::execute_text_request_for_testing(
            &mut session,
            &format!("addpath('{unrelated_path}'); answer = ordinary_add_one(45); answer"),
        )
        .unwrap()
    };
    assert!(unchanged.error.is_none(), "{:?}", unchanged.error);
    assert_eq!(unchanged.value, Some(Value::Num(49.0)));

    // Loaded code is pinned to its validated private bytes until a verified
    // clear. Mutation of the original is not an unverified code replacement.
    std::fs::write(&artifact.module, b"changed after admission").unwrap();
    let pinned = {
        let _entered = runtime.enter();
        runmat_core::execute_text_request_for_testing(
            &mut session,
            "answer = ordinary_add_one(43); answer",
        )
        .unwrap()
    };
    assert!(pinned.error.is_none(), "{:?}", pinned.error);
    assert_eq!(pinned.value, Some(Value::Num(48.0)));
    let pinned_helper = {
        let _entered = runtime.enter();
        runmat_core::execute_text_request_for_testing(
            &mut session,
            "answer = ordinary_mex_helper(45); answer",
        )
        .unwrap()
    };
    assert!(pinned_helper.error.is_none(), "{:?}", pinned_helper.error);
    assert_eq!(pinned_helper.value, Some(Value::Num(51.0)));
    let cleared = {
        let _entered = runtime.enter();
        runmat_core::execute_text_request_for_testing(&mut session, "clear ordinary_add_one;")
            .unwrap()
    };
    assert!(cleared.error.is_none(), "{:?}", cleared.error);
    let rejected_helper = {
        let _entered = runtime.enter();
        runmat_core::execute_text_request_for_testing(
            &mut session,
            "answer = ordinary_mex_helper(45); answer",
        )
        .unwrap()
    };
    assert_eq!(
        rejected_helper
            .error
            .as_ref()
            .and_then(|error| error.identifier()),
        Some("RunMat:MEX:Artifact")
    );

    // Never-admitted bytes must also fail before native load.
    let never_source = fixture.path().join("ordinary_never_admitted.c");
    std::fs::write(&never_source, std::fs::read_to_string(&source).unwrap()).unwrap();
    let never = runmat_mex::MexBuild::new(&never_source, fixture.path())
        .compile()
        .unwrap();
    std::fs::write(&never.module, b"changed before admission").unwrap();
    let rejected_first = {
        let _entered = runtime.enter();
        runmat_core::execute_text_request_for_testing(
            &mut session,
            "answer = ordinary_never_admitted(1); answer",
        )
        .unwrap()
    };
    assert_eq!(
        rejected_first
            .error
            .as_ref()
            .and_then(|error| error.identifier()),
        Some("RunMat:MEX:Artifact")
    );
}

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
            "[package]\nname = \"project-mex\"\n[sources]\nroots = [\"src\"]\n[mex-artifacts.project_fixture]\nmanifest = {}\nmodule = {}\n",
            serde_json::to_string(&manifest.to_string_lossy().replace('\\', "/")).unwrap(),
            serde_json::to_string(&module.to_string_lossy().replace('\\', "/")).unwrap()
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
    let frozen_artifact = handoff
        .project
        .mex_artifacts
        .iter()
        .find(|entry| entry.name == "project_fixture")
        .unwrap();
    let manifest_path = &frozen_artifact.manifest_path;
    let expected_digest = &frozen_artifact.manifest_digest;
    assert_eq!(
        expected_digest,
        &runmat_package::ContentDigest::sha256(&manifest_bytes)
    );
    let changed_manifest = b"invalid MEX manifest";
    std::fs::write(&artifact.manifest, changed_manifest).unwrap();
    let rejected = runtime
        .block_on(session.install_project_handoff(handoff.clone()))
        .expect_err("invalid replacement must not retire the live project");
    let runmat_package::FrozenProjectHandoffError::Revision(message) = &rejected else {
        panic!("expected the exact frozen artifact digest refusal: {rejected}");
    };
    assert!(message.contains(&format!(
        "foreign artifact {} changed after project resolution: expected {}, found {}",
        manifest_path.display(),
        expected_digest,
        runmat_package::ContentDigest::sha256(changed_manifest),
    )));
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
