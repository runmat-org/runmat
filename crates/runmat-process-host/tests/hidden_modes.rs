use std::ffi::OsString;

use runmat_process_host::{HiddenMode, HiddenModeRegistry};

fn args(values: &[&str]) -> Vec<OsString> {
    values.iter().map(OsString::from).collect()
}

#[test]
fn registry_recognizes_exact_private_mode() {
    let mode = HiddenModeRegistry::standard()
        .detect(args(&["runmat", "--__runmat-test-worker"]))
        .unwrap();
    assert_eq!(mode, Some(HiddenMode::TestWorker));
}

#[test]
fn registry_recognizes_meshing_worker_mode() {
    let mode = HiddenModeRegistry::standard()
        .detect(args(&["runmat", "--__runmat-meshing-worker"]))
        .unwrap();
    assert_eq!(mode, Some(HiddenMode::MeshingWorker));
}

#[test]
fn registry_rejects_private_mode_mixed_with_user_arguments() {
    let error = HiddenModeRegistry::standard()
        .detect(args(&["runmat", "--__runmat-test-worker", "--help"]))
        .unwrap_err();
    assert!(error.to_string().contains("sole process argument"));
}

#[test]
fn ordinary_arguments_do_not_select_a_private_mode() {
    let mode = HiddenModeRegistry::standard()
        .detect(args(&["runmat", "script.m"]))
        .unwrap();
    assert_eq!(mode, None);
}
