use super::test_arguments::h5write_builtin;
use futures::executor::block_on;
use runmat_value::Value;
use tempfile::tempdir;

use super::*;

#[test]
fn invalid_string_encoding_precedes_file_creation_and_attribute_replacement() {
    let _lock = runmat_filesystem::provider_override_lock();
    let dir = tempdir().expect("isolated fixture");
    let missing = dir.path().join("missing.h5");
    let error = block_on(h5write_builtin(
        None,
        Value::String(missing.display().to_string().into()),
        Value::String("/A".into()),
        Value::String("bad\0value".into()),
        Vec::new(),
    ))
    .expect_err("embedded NUL is invalid HDF5 variable-length text");
    assert_eq!(error.identifier(), ERR_ARGUMENT.identifier);
    assert!(
        !missing.exists(),
        "invalid input must not create the HDF5 file"
    );

    let existing = Value::String(dir.path().join("existing.h5").display().to_string().into());
    block_on(h5write_builtin(
        None,
        existing.clone(),
        Value::String("/A".into()),
        Value::Num(1.0),
        Vec::new(),
    ))
    .expect("dataset setup");
    block_on(h5writeatt_builtin(
        None,
        existing.clone(),
        Value::String("/A".into()),
        Value::String("units".into()),
        Value::String("m/s".into()),
    ))
    .expect("attribute setup");
    let error = block_on(h5writeatt_builtin(
        None,
        existing.clone(),
        Value::String("/A".into()),
        Value::String("units".into()),
        Value::String("bad\0value".into()),
    ))
    .expect_err("invalid replacement must not delete old attribute");
    assert_eq!(error.identifier(), ERR_ARGUMENT.identifier);

    let metadata = block_on(h5info_builtin(
        existing,
        crate::builtins::common::test_support::builtin_arguments(vec![Value::String("/A".into())]),
    ))
    .expect("read dataset metadata");
    let Value::Struct(metadata) = metadata else {
        panic!("dataset metadata struct")
    };
    let Some(Value::Cell(attributes)) = metadata.fields.get("Attributes") else {
        panic!("attribute cell")
    };
    assert_eq!(attributes.data().len(), 1);
}
