//! Cheap public-entrypoint regression tests; no numerical solve is started.
use std::{fs, process::Command};
#[path = "support/local_offline.rs"]
mod local_offline;

#[test]
fn plain_script_needs_no_session_or_server() {
    let project = tempfile::tempdir().unwrap();
    fs::write(
        project.path().join("plain.m"),
        "if false\n  unused = fea.run(\"not-executed.fea\");\nend\ndisp(7);\n",
    )
    .unwrap();
    for disable_jit in [false, true] {
        let mut command = Command::new(env!("CARGO_BIN_EXE_runmat"));
        local_offline::configure(&mut command, project.path());
        if disable_jit {
            command.arg("--no-jit");
        }
        let output = command
            .current_dir(project.path())
            .arg("plain.m")
            .env("RUNMAT_ORG_ID", "invalid-local-test-selector")
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "disable_jit={disable_jit}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(String::from_utf8_lossy(&output.stdout).contains('7'));
        assert!(!project.path().join("signed-out-cli/remote.json").exists());
    }
}

#[test]
#[cfg(feature = "occt-native")]
fn signed_out_local_fea_is_denied_before_meshing() {
    let project = tempfile::tempdir().unwrap();
    fs::create_dir_all(project.path().join("signed-out-cli")).unwrap();
    fs::write(
        project.path().join("signed-out-cli/remote.json"),
        r#"{"credential_store":"file","server_url":"http://127.0.0.1:0"}"#,
    )
    .unwrap();
    fs::write(
        project.path().join("study.fea"),
        r#"
geometry part {
    construct {
        feature body = box(width = 1 m, depth = 0.5 m, height = 0.25 m)
        result = feature.body
    }
}
mesh volume on geometry.part {
    tetrahedra {
        order = linear
        target_size = 0.1 m
        maximum_deviation = 0.01 m
    }
}
model structure {}
study static {
    geometry = geometry.part
    mesh = mesh.volume
    model = model.structure
    analysis solve {
        structural static {}
    }
    output displacement = analysis.solve.field.node.displacement
}
"#,
    )
    .unwrap();
    let mut command = Command::new(env!("CARGO_BIN_EXE_runmat"));
    local_offline::configure(&mut command, project.path());
    let output = command
        .current_dir(project.path())
        .args(["run", "study.fea", "--json"])
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!output.status.success());
    assert!(stderr.contains("RM.FEA.SignInRequired"), "{stderr}");
    assert!(
        stderr.contains("Sign in with `runmat login` to mesh or solve."),
        "{stderr}"
    );
    let mut invalid = Command::new(env!("CARGO_BIN_EXE_runmat"));
    local_offline::configure(&mut invalid, project.path());
    let output = invalid
        .current_dir(project.path())
        .args(["run", "study.fea", "--json"])
        .env("RUNMAT_ORG_ID", "invalid-local-test-selector")
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("RM.FEA.LeaseInvalid"));
}
