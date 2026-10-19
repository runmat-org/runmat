//! Explicit developer diagnostic extraction through the production bounded reader.
//! This exports an unaccepted input, never a mesh or a qualification receipt.
use super::*;

#[test]
#[ignore = "requires an explicit retained diagnostic key, cache root and new output path"]
fn meshing_export_retained_quality_diagnostic() {
    let text = std::env::var("RUNMAT_DIAGNOSTIC_MANIFEST_KEY").unwrap();
    assert_eq!(text.len(), 64);
    let bytes = std::array::from_fn(|index| {
        u8::from_str_radix(&text[index * 2..index * 2 + 2], 16).unwrap()
    });
    let key = runmat_meshing_core::StableDigest::from_bytes(bytes);
    let output = std::env::var_os("RUNMAT_DIAGNOSTIC_OUTPUT").unwrap();
    let control = Control {
        limit: Some(512 * 1024 * 1024),
        ..Control::default()
    };
    let store = crate::meshing_subtask_cache::native_subtask_cache().unwrap();
    assert!(store.admit(&control).unwrap());
    let connection_storage = control.live.load(Ordering::Relaxed);
    let record = runmat_meshing_execution::read_volume_diagnostic(store.as_ref(), key, &control)
        .unwrap()
        .expect("retained diagnostic missing or evicted");
    use std::io::Write;
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)
        .unwrap();
    file.write_all(record.bytes()).unwrap();
    file.sync_all().unwrap();
    eprintln!(
        "Unaccepted quality diagnostic exported: {} bytes, manifest {text}",
        record.bytes().len()
    );
    drop(record);
    assert_eq!(control.live.load(Ordering::Relaxed), connection_storage);
}
