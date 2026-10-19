use super::*;
use runmat_filesystem::{self as fs, FsProvider, MemoryFsProvider};
use std::path::Path;
use std::sync::Arc;

#[test]
fn staged_write_keeps_captured_provider_and_working_directory() {
    let _serial = fs::provider_override_lock();
    let first = Arc::new(MemoryFsProvider::with_current_dir("/first"));
    let second = Arc::new(MemoryFsProvider::with_current_dir("/second"));
    let _selected = fs::replace_provider(first.clone() as Arc<dyn FsProvider>);
    let path = FigureFsPath::capture(Path::new("scene.fig")).expect("capture first path");
    assert_eq!(path.path(), Path::new("/first/scene.fig"));
    assert_eq!(
        path.provider_identity(),
        Arc::as_ptr(&(first.clone() as Arc<dyn FsProvider>)) as *const () as usize
    );
    fs::set_provider(second.clone() as Arc<dyn FsProvider>);

    block_on(write_bytes(None, &path, b"scene", SAVEFIG)).expect("publish on captured provider");
    assert_eq!(
        block_on(first.read(Path::new("/first/scene.fig"))).unwrap(),
        b"scene"
    );
    assert!(block_on(second.read(Path::new("/second/scene.fig"))).is_err());
}
