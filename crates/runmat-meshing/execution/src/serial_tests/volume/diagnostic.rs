//! Opt-in retention of a failed short fixture, never an accepted mesh artifact.
pub(super) fn retain_failure(
    store: &crate::subtask_cache::tests::Store,
    request: &runmat_meshing_core::MeshingRequest,
) {
    let Some(path) = std::env::var_os("RUNMAT_TEST_VOLUME_FAILURE_DIR") else {
        return;
    };
    let path = std::path::PathBuf::from(path);
    std::fs::create_dir_all(&path).unwrap();
    let records = store.0.lock().unwrap();
    assert!(records.len() <= 1024);
    assert!(records.values().map(|(b, _)| b.len()).sum::<usize>() <= 16 * 1024 * 1024);
    for (key, (bytes, _)) in records.iter() {
        let name = key
            .bytes()
            .iter()
            .map(|v| format!("{v:02x}"))
            .collect::<String>();
        std::fs::write(path.join(name), bytes).unwrap();
    }
    let options = crate::volume_kernel::volume_options(request);
    std::fs::write(
        path.join("options.json"),
        serde_json::to_vec_pretty(&options).unwrap(),
    )
    .unwrap();
}
