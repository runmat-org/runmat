#![cfg(not(target_family = "wasm"))]

use std::process::Command;

use runmat_native_ffi::NativeInterfaceArtifactManifest;

fn clang_available() -> bool {
    Command::new("clang")
        .arg("--version")
        .output()
        .is_ok_and(|output| output.status.success())
}

#[test]
fn command_prepares_the_canonical_exact_library_artifact() {
    if !clang_available() {
        eprintln!("skipping native-interface CLI test because clang is unavailable");
        return;
    }
    let temporary = tempfile::tempdir().unwrap();
    let library = temporary.path().join("libfixture.bin");
    let header = temporary.path().join("fixture.h");
    let output = temporary.path().join("artifacts/fixture.json");
    std::fs::write(&library, b"native library fixture bytes").unwrap();
    std::fs::write(
        &header,
        "#ifdef FEATURE_ENABLED\nint fixture_value(int input);\n#endif\n",
    )
    .unwrap();

    let completed = Command::new(env!("CARGO_BIN_EXE_runmat"))
        .args(["--color=never", "native-interface", "prepare"])
        .arg("--library")
        .arg(&library)
        .args(["--library-name", "fixture_binary", "--header"])
        .arg(&header)
        .args([
            "--interface-name",
            "fixture",
            "-DFEATURE_ENABLED=1",
            "--output",
        ])
        .arg(&output)
        .arg("--json")
        .env("NO_GUI", "1")
        .output()
        .unwrap();
    assert!(
        completed.status.success(),
        "stdout={}\nstderr={}",
        String::from_utf8_lossy(&completed.stdout),
        String::from_utf8_lossy(&completed.stderr)
    );
    let report: serde_json::Value = serde_json::from_slice(&completed.stdout).unwrap();
    assert_eq!(report["interface_name"], "fixture");
    assert_eq!(report["library_name"], "fixture_binary");
    assert_eq!(report["manifest"], output.display().to_string());

    let manifest = NativeInterfaceArtifactManifest::read(&output).unwrap();
    manifest
        .validate_current_library(&std::fs::read(library).unwrap())
        .unwrap();
    assert_eq!(
        manifest.metadata.libraries[0].symbols[0].name,
        "fixture_value"
    );
}
