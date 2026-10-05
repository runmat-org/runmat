use super::*;
use runmat_filesystem::MemoryFsProvider;
use std::path::Path;
use std::sync::Arc;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn resize_and_overlay_transfer_owned_text_without_cloning_payloads() {
    let _lock = runmat_filesystem::provider_override_lock();
    let provider = Arc::new(MemoryFsProvider::with_current_dir("/xlswrite-overlay"));
    let _restore = runmat_filesystem::replace_provider(provider);
    let selected = TabularFsOwner::capture(Some(Path::new("/xlswrite-overlay/out.xlsx")))
        .expect("captured owner")
        .path("/xlswrite-overlay/out.xlsx".into())
        .expect("selected path");
    let old = "o".repeat(8192);
    let old_ptr = old.as_ptr();
    let new = "n".repeat(8192);
    let new_ptr = new.as_ptr();
    let target = XlsTable {
        rows: 1,
        cols: 1,
        cells: vec![CellValue::Text(old)],
    };
    let incoming = XlsTable {
        rows: 1,
        cols: 1,
        cells: vec![CellValue::Text(new)],
    };
    let mut cooperation = TabularCooperation::new(None, BUILTIN_NAME);
    let target = futures::executor::block_on(target.overlay(
        RangeStart { row: 1, col: 1 },
        incoming,
        &mut cooperation,
        &selected,
    ))
    .expect("owned overlay");
    assert_eq!((target.rows, target.cols), (2, 2));
    assert!(matches!(&target.cells[0], CellValue::Text(text) if text.as_ptr() == old_ptr));
    assert!(matches!(&target.cells[3], CellValue::Text(text) if text.as_ptr() == new_ptr));
}
