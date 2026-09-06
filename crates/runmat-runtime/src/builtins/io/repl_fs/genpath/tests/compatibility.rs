use runmat_value::{IntegerStorage, Tensor, Value};
use tempfile::tempdir;

use super::support::{call, canonical, text};

#[test]
fn matlab_mode_rejects_the_excludes_extension() {
    let _compatibility = crate::compatibility::push_runmat_extensions_enabled(false);
    let error = call(vec![
        Value::String("root".into()),
        Value::String("skip".into()),
    ])
    .expect_err("excludes extension");
    assert_eq!(
        error.identifier(),
        Some("RunMat:compatibility:GenpathExcludesExtension")
    );
}

#[test]
fn all_integer_classes_are_exact_and_mode_gated() {
    let directory = tempdir().expect("directory");
    let codes: Vec<u32> = directory
        .path()
        .to_string_lossy()
        .chars()
        .map(u32::from)
        .collect();
    let storages = [
        IntegerStorage::I8(codes.iter().map(|value| *value as i8).collect()),
        IntegerStorage::I16(codes.iter().map(|value| *value as i16).collect()),
        IntegerStorage::I32(codes.iter().map(|value| *value as i32).collect()),
        IntegerStorage::I64(codes.iter().map(|value| i64::from(*value)).collect()),
        IntegerStorage::U8(codes.iter().map(|value| *value as u8).collect()),
        IntegerStorage::U16(codes.iter().map(|value| *value as u16).collect()),
        IntegerStorage::U32(codes.clone()),
        IntegerStorage::U64(codes.iter().map(|value| u64::from(*value)).collect()),
    ];

    let strict_value = Value::Tensor(
        Tensor::new_integer(storages[0].clone(), vec![1, codes.len()]).expect("codes"),
    );
    let strict = crate::compatibility::push_runmat_extensions_enabled(false);
    let error = call(vec![strict_value]).expect_err("numeric extension");
    assert_eq!(
        error.identifier(),
        Some("RunMat:compatibility:GenpathNumericCharacterCodesExtension")
    );
    drop(strict);

    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    for storage in storages {
        let value =
            Value::Tensor(Tensor::new_integer(storage, vec![1, codes.len()]).expect("codes"));
        assert_eq!(
            text(call(vec![value]).expect("genpath")),
            canonical(directory.path())
        );
    }
}

#[test]
fn validates_the_whole_call_before_resident_gather() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let resident = Value::GpuTensor(runmat_accelerate_api::GpuTensorHandle {
        shape: vec![1, 1],
        device_id: 0,
        buffer_id: 9_343_101,
        descriptor: Default::default(),
    });
    let error = call(vec![resident, Value::Bool(true)]).expect_err("invalid excludes");
    assert_eq!(
        error.identifier(),
        runmat_builtins::GENPATH_ERROR_EXCLUDES_TYPE.identifier
    );
    assert!(!error.message().to_ascii_lowercase().contains("provider"));
}

#[test]
fn matlab_mode_rejects_resident_numeric_before_provider_lookup() {
    let _compatibility = crate::compatibility::push_runmat_extensions_enabled(false);
    let resident = Value::GpuTensor(runmat_accelerate_api::GpuTensorHandle {
        shape: vec![1, 1],
        device_id: 0,
        buffer_id: 9_343_102,
        descriptor: Default::default(),
    });
    let error = call(vec![resident]).expect_err("numeric extension");
    assert_eq!(
        error.identifier(),
        Some("RunMat:compatibility:GenpathNumericCharacterCodesExtension")
    );
}
