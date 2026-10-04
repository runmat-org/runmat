use super::*;
use crate::{FsProvider, MemoryFsProvider};
use futures::executor::block_on;
use std::io::ErrorKind;
use std::path::Path;
use std::sync::Arc;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn memory_backing_tracks_live_in_place_bytes_and_rejects_replacement() {
    let _lock = crate::provider_override_lock();
    let first = Arc::new(MemoryFsProvider::with_current_dir("/mapped"));
    first
        .write_project_path("/mapped/data.bin", b"abcdef")
        .unwrap();
    let _selected = crate::replace_provider(first.clone());
    crate::set_current_dir("/mapped").unwrap();
    let mut backing = block_on(capture_mapped_backing("data.bin", true))
        .unwrap()
        .expect("memory provider supports opened ranges");
    assert_eq!(backing.path(), Path::new("/mapped/data.bin"));
    assert_eq!(backing.opened_extent(), 6);

    let second = Arc::new(MemoryFsProvider::with_current_dir("/other"));
    second
        .write_project_path("/other/data.bin", b"wrong!")
        .unwrap();
    let _replacement = crate::replace_provider(second.clone());
    crate::set_current_dir("/other").unwrap();
    assert_eq!(block_on(backing.write_at(2, b"XY")).unwrap(), 2);
    assert_eq!(
        first.read_project_path("/mapped/data.bin").unwrap(),
        b"abXYef"
    );
    assert_eq!(
        second.read_project_path("/other/data.bin").unwrap(),
        b"wrong!"
    );

    first
        .write_project_path("/mapped/data.bin", b"123456")
        .unwrap();
    let mut read = [0; 3];
    assert_eq!(block_on(backing.read_at(1, &mut read)).unwrap(), 3);
    assert_eq!(&read, b"234");
    assert_eq!(
        block_on(backing.write_at(5, b"too long"))
            .unwrap_err()
            .kind(),
        ErrorKind::InvalidInput
    );
    assert_eq!(
        first.read_project_path("/mapped/data.bin").unwrap(),
        b"123456"
    );

    block_on(first.remove_file(Path::new("/mapped/data.bin"))).unwrap();
    first
        .write_project_path("/mapped/data.bin", b"reused")
        .unwrap();
    assert_eq!(
        backing.check_current().unwrap_err().kind(),
        ErrorKind::PermissionDenied
    );
    assert_eq!(
        block_on(backing.write_at(0, b"X")).unwrap_err().kind(),
        ErrorKind::PermissionDenied
    );
    assert_eq!(
        first.read_project_path("/mapped/data.bin").unwrap(),
        b"reused"
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn memory_readonly_and_truncation_do_not_extend_opened_backing() {
    let provider = MemoryFsProvider::with_current_dir("/mapped");
    provider
        .write_project_path("/mapped/data.bin", b"abcd")
        .unwrap();
    let mut backing = block_on(provider.open_mapped_backing(Path::new("/mapped/data.bin"), false))
        .unwrap()
        .expect("memory range backing");
    assert!(!backing.writable());
    assert_eq!(
        block_on(backing.write_at(0, b"X")).unwrap_err().kind(),
        ErrorKind::PermissionDenied
    );
    provider
        .write_project_path("/mapped/data.bin", b"ab")
        .unwrap();
    let mut read = [0; 4];
    assert_eq!(block_on(backing.read_at(0, &mut read)).unwrap(), 2);
    assert_eq!(&read[..2], b"ab");

    let mut writer = block_on(provider.open_mapped_backing(Path::new("/mapped/data.bin"), true))
        .unwrap()
        .expect("memory writable range backing");
    provider
        .write_project_path("/mapped/data.bin", b"a")
        .unwrap();
    assert_eq!(
        block_on(writer.write_at(1, b"Z")).unwrap_err().kind(),
        ErrorKind::UnexpectedEof
    );
    assert_eq!(
        provider.read_project_path("/mapped/data.bin").unwrap(),
        b"a"
    );
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn native_backing_writes_only_inside_its_opened_extent() {
    use crate::NativeFsProvider;
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("data.bin");
    std::fs::write(&path, b"abcd").unwrap();
    let provider = NativeFsProvider;
    let mut backing = block_on(provider.open_mapped_backing(&path, true))
        .unwrap()
        .expect("native opened range backing");
    assert_eq!(backing.opened_extent(), 4);
    assert_eq!(block_on(backing.write_at(1, b"XY")).unwrap(), 2);
    assert_eq!(std::fs::read(&path).unwrap(), b"aXYd");
    assert_eq!(
        block_on(backing.write_at(4, b"Z")).unwrap_err().kind(),
        ErrorKind::InvalidInput
    );
    assert_eq!(std::fs::metadata(&path).unwrap().len(), 4);
}

#[cfg(unix)]
#[test]
fn native_backing_pins_opened_file_and_exposes_external_in_place_edits() {
    use crate::NativeFsProvider;
    let root = tempfile::tempdir().unwrap();
    let original = root.path().join("data.bin");
    let moved = root.path().join("moved.bin");
    std::fs::write(&original, b"abcdef").unwrap();
    let provider = NativeFsProvider;
    let mut backing = block_on(provider.open_mapped_backing(&original, true))
        .unwrap()
        .expect("native opened range backing");
    std::fs::rename(&original, &moved).unwrap();
    std::fs::write(&original, b"wrong!").unwrap();
    assert_eq!(block_on(backing.write_at(2, b"XY")).unwrap(), 2);
    assert_eq!(std::fs::read(&moved).unwrap(), b"abXYef");
    assert_eq!(std::fs::read(&original).unwrap(), b"wrong!");
    std::fs::write(&moved, b"123456").unwrap();
    let mut read = [0; 3];
    assert_eq!(block_on(backing.read_at(1, &mut read)).unwrap(), 3);
    assert_eq!(&read, b"234");
    assert_eq!(
        block_on(backing.write_at(6, b"X")).unwrap_err().kind(),
        ErrorKind::InvalidInput
    );
    std::fs::write(&moved, b"a").unwrap();
    assert_eq!(
        block_on(backing.write_at(0, b"X")).unwrap_err().kind(),
        ErrorKind::UnexpectedEof
    );
}
