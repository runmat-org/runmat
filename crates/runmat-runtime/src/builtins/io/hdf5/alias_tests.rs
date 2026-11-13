use super::test_arguments::{hdf5read_builtin, hdf5write_builtin};
use futures::executor::block_on;
use runmat_value::{IntegerStorage, Tensor, Value};
use tempfile::tempdir;

use super::*;

#[test]
fn legacy_read_write_and_info_share_exact_native_dataset_semantics() {
    let _fs_lock = runmat_filesystem::provider_override_lock();
    let dir = tempdir().expect("isolated HDF5 fixture");
    let filename = Value::String(dir.path().join("aliases.h5").display().to_string().into());
    let data = Value::Tensor(
        Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX, 1_u64 << 63]), vec![1, 2])
            .expect("exact uint64 tensor"),
    );
    let written = block_on(hdf5write_builtin(
        None,
        filename.clone(),
        Value::String("/group/A".into()),
        data,
        Vec::new(),
    ))
    .expect("legacy write");
    assert!(matches!(written, Value::OutputList(outputs) if outputs.is_empty()));

    let read = block_on(hdf5read_builtin(
        None,
        filename.clone(),
        Value::String("/group/A".into()),
        Vec::new(),
    ))
    .expect("legacy read");
    let Value::Tensor(read) = read else {
        panic!("expected typed tensor")
    };
    assert_eq!(read.shape(), &[1, 2]);
    assert_eq!(
        read.integer_storage(),
        Some(&IntegerStorage::U64(vec![u64::MAX, 1_u64 << 63]))
    );

    let info = block_on(hdf5info_builtin(
        filename,
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .expect("legacy metadata");
    let Value::Struct(info) = info else {
        panic!("expected metadata struct")
    };
    assert_eq!(info.fields.get("Name"), Some(&Value::String("/".into())));
    assert!(info.fields.contains_key("Groups"));
}
