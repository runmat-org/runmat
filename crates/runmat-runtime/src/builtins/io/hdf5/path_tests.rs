use super::test_arguments::{h5read_builtin, h5write_builtin};
use std::sync::Arc;

use futures::executor::block_on;
use runmat_filesystem::{replace_provider, MemoryFsProvider};
use runmat_value::Value;
use tempfile::tempdir;

use super::*;

#[test]
fn virtual_filesystem_cannot_fall_through_to_ambient_native_hdf5_file() {
    let _lock = runmat_filesystem::provider_override_lock();
    let root = tempdir().expect("native temp directory");
    let path = root.path().join("must-not-create.h5");
    let _provider = replace_provider(Arc::new(MemoryFsProvider::with_current_dir("/virtual")));

    let write = block_on(h5write_builtin(
        None,
        Value::String(path.display().to_string().into()),
        Value::String("/data".into()),
        Value::Num(1.0),
        Vec::new(),
    ))
    .expect_err("memory provider has no native path grant");
    let expected = if cfg!(windows) {
        ERR_IO.identifier
    } else {
        ERR_PROVIDER.identifier
    };
    assert_eq!(write.identifier(), expected);
    assert!(!path.exists());

    let read = block_on(h5read_builtin(
        None,
        Value::String(path.display().to_string().into()),
        Value::String("/data".into()),
        Vec::new(),
    ))
    .expect_err("read cannot use ambient native path");
    assert_eq!(read.identifier(), expected);
    // A Windows drive-prefixed path is outside the virtual rooted namespace;
    // capture rejects that spelling before asking for a native-path grant.
    #[cfg(windows)]
    for error in [&write, &read] {
        assert_eq!(
            error
                .source
                .as_deref()
                .and_then(|cause| cause.downcast_ref::<std::io::Error>())
                .map(std::io::Error::kind),
            Some(std::io::ErrorKind::InvalidInput)
        );
    }
    assert!(!path.exists());
}
