use runmat_value::{IntegerStorage, Tensor, Value};
use tempfile::tempdir;

use super::super::super::path_mutation::test_support::{canonical, PathGuard};

#[test]
fn all_integer_classes_are_exact_and_mode_gated() {
    let _guard = PathGuard::new();
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

    let value = Value::Tensor(
        Tensor::new_integer(storages[0].clone(), vec![1, codes.len()]).expect("codes"),
    );
    let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
    let error = super::run(vec![value]).expect_err("compatibility rejection");
    assert_eq!(
        error.identifier(),
        Some("RunMat:compatibility:AddpathNumericCharacterCodesExtension")
    );
    drop(_compat);

    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    for storage in storages {
        crate::builtins::common::path_state::set_path_string("");
        let value =
            Value::Tensor(Tensor::new_integer(storage, vec![1, codes.len()]).expect("codes"));
        super::run(vec![value]).expect("RunMat extension");
        assert_eq!(
            crate::builtins::common::path_state::current_path_segments(),
            vec![canonical(directory.path())]
        );
    }
}

#[test]
fn resident_numeric_rejects_before_provider_lookup_in_matlab_mode() {
    let _guard = PathGuard::new();
    let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
    let value = Value::GpuTensor(runmat_accelerate_api::GpuTensorHandle {
        shape: vec![1, 1],
        device_id: 0,
        buffer_id: 9_343_001,
        descriptor: Default::default(),
    });
    let error = super::run(vec![value]).expect_err("compatibility rejection");
    assert_eq!(
        error.identifier(),
        Some("RunMat:compatibility:AddpathNumericCharacterCodesExtension")
    );
}

#[test]
fn validates_all_arguments_before_gathering_resident_character_codes() {
    let _guard = PathGuard::new();
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let resident = Value::GpuTensor(runmat_accelerate_api::GpuTensorHandle {
        shape: vec![1, 1],
        device_id: 0,
        buffer_id: 9_343_002,
        descriptor: Default::default(),
    });
    let error = super::run(vec![resident, Value::Bool(true)]).expect_err("invalid argument");
    assert_eq!(
        error.identifier(),
        runmat_builtins::ADDPATH_ERROR_ARGUMENT_TYPE.identifier
    );
    assert!(!error.message().to_ascii_lowercase().contains("provider"));
}
