use runmat_types::IntegerClass;
use runmat_value::{NumericStorage, Tensor, Value};
use tempfile::tempdir;

#[test]
fn strict_mode_rejects_each_runmat_only_surface() {
    let _lock = super::support::lock();
    let directory = tempdir().expect("directory");
    let diagnostic = super::support::run(Vec::new(), 3, false).expect_err("diagnostics");
    assert_eq!(
        diagnostic.identifier(),
        runmat_builtins::SAVEPATH_DIAGNOSTIC_OUTPUTS_EXTENSION.error_identifier
    );

    let directory_target = super::support::run(
        vec![Value::String(
            directory.path().to_string_lossy().into_owned(),
        )],
        1,
        false,
    )
    .expect_err("directory target");
    assert_eq!(
        directory_target.identifier(),
        runmat_builtins::SAVEPATH_DIRECTORY_TARGET_EXTENSION.error_identifier
    );

    let numeric = Tensor::new(vec![112.0], vec![1, 1]).expect("numeric");
    let numeric_error =
        super::support::run(vec![Value::Tensor(numeric)], 1, false).expect_err("numeric filename");
    assert_eq!(
        numeric_error.identifier(),
        runmat_builtins::SAVEPATH_NUMERIC_CHARACTER_CODES_EXTENSION.error_identifier
    );
}

#[test]
fn runmat_mode_accepts_all_integer_filename_classes() {
    let _lock = super::support::lock();
    let directory = tempdir().expect("directory");
    for class in [
        IntegerClass::Int8,
        IntegerClass::Int16,
        IntegerClass::Int32,
        IntegerClass::Int64,
        IntegerClass::UInt8,
        IntegerClass::UInt16,
        IntegerClass::UInt32,
        IntegerClass::UInt64,
    ] {
        let target = directory.path().join(format!("{}.m", class.class_name()));
        let codes: Vec<u32> = target
            .to_string_lossy()
            .chars()
            .map(|character| character as u32)
            .collect();
        let storage = storage(class, &codes);
        let tensor = Tensor::from_numeric_storage(storage, vec![1, codes.len()]).expect("tensor");
        let value =
            super::support::run(vec![Value::Tensor(tensor)], 1, true).expect("integer filename");
        assert_eq!(super::support::status(&value), 0.0, "{class:?}");
        assert!(target.exists(), "{class:?}");
    }
}

fn storage(class: IntegerClass, codes: &[u32]) -> NumericStorage {
    match class {
        IntegerClass::Int8 => NumericStorage::I8(codes.iter().map(|&value| value as i8).collect()),
        IntegerClass::Int16 => {
            NumericStorage::I16(codes.iter().map(|&value| value as i16).collect())
        }
        IntegerClass::Int32 => {
            NumericStorage::I32(codes.iter().map(|&value| value as i32).collect())
        }
        IntegerClass::Int64 => {
            NumericStorage::I64(codes.iter().map(|&value| value as i64).collect())
        }
        IntegerClass::UInt8 => NumericStorage::U8(codes.iter().map(|&value| value as u8).collect()),
        IntegerClass::UInt16 => {
            NumericStorage::U16(codes.iter().map(|&value| value as u16).collect())
        }
        IntegerClass::UInt32 => NumericStorage::U32(codes.to_vec()),
        IntegerClass::UInt64 => {
            NumericStorage::U64(codes.iter().map(|&value| u64::from(value)).collect())
        }
    }
}

#[test]
fn strict_mode_does_not_create_a_missing_parent() {
    let _lock = super::support::lock();
    let directory = tempdir().expect("directory");
    let target = directory.path().join("missing").join("pathdef.m");
    let strict = super::support::run(
        vec![Value::String(target.to_string_lossy().into_owned())],
        1,
        false,
    )
    .expect("strict status");
    assert_eq!(super::support::status(&strict), 1.0);
    assert!(!target.exists());

    let runmat = super::support::run(
        vec![Value::String(target.to_string_lossy().into_owned())],
        1,
        true,
    )
    .expect("runmat status");
    assert_eq!(super::support::status(&runmat), 0.0);
    assert!(target.exists());
}
